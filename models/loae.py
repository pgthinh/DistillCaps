import random

import torch
import torch.nn.functional as F
from models.base_captioning import BaseCaptioning
from models.ced.audiotransformer import AudioTransformer, CEDConfig
from peft import LoraConfig, TaskType
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoFeatureExtractor, ClapAudioModel
from torch import nn
from torch.nn import CrossEntropyLoss

class FFTAdapter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.freq_transform = nn.Linear(embed_dim, embed_dim)  # Learnable transform in frequency space

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim) – input audio token embeddings
        """
        # Step 1: Apply FFT along the sequence dimension
        x_freq = torch.fft.fft(x, dim=1)  # FFT over time (seq_len axis)
        
        # Step 2: Transform frequency components
        x_freq = self.freq_transform(x_freq.real) - 1j * self.freq_transform(x_freq.imag)  # Linear mapping in frequency space
        
        # Step 3: Apply inverse FFT to return to time domain
        x_time = torch.fft.ifft(x_freq, dim=1).real  # Take real part
        
        return x_time

def average_similarity(
    align_image_features: torch.Tensor,
    align_input_embeds: torch.Tensor,
    temperature: torch.Tensor,
):

    align_image_features_pool = align_image_features.mean(dim=1)

    align_image_features_pool = (
        align_image_features_pool / align_image_features_pool.norm(dim=1)[:, None]
    )

    align_input_embeds_pool = align_input_embeds.mean(dim=1)
    align_input_embeds_pool = (
        align_input_embeds_pool / align_input_embeds_pool.norm(dim=1)[:, None]
    )
    matrixSimi = torch.exp(temperature) * torch.mm(align_image_features_pool, align_input_embeds_pool.transpose(0, 1))

    return matrixSimi

def compute_loss(
    contrastive_loss_type: str,
    matrixSimi: torch.Tensor,
    beta=None,
):
    if contrastive_loss_type == "infonce":
        targetAlign = torch.arange(matrixSimi.shape[0]).to(
            matrixSimi.device
        )
        lossAlign = (CrossEntropyLoss()(matrixSimi, targetAlign) + CrossEntropyLoss()(matrixSimi.T, targetAlign))/2.0
    elif contrastive_loss_type == "siglip":
        matrixSimi += beta
        targetAlign = (
            2 * torch.eye(matrixSimi.shape[0])
            - torch.ones(matrixSimi.shape[0])
        ).to(matrixSimi.device)
        lossAlign = (
            -1.0
            * torch.sum(torch.nn.LogSigmoid()(matrixSimi * targetAlign))
            / matrixSimi.shape[0]
        )
    else:
        raise ValueError("Wrong contrastive loss type.")

    return lossAlign
   
class CedLlama7BCaptionModel(BaseCaptioning):
    def __init__(self, config):
        super().__init__(config)        

        # the checkpoint can be downloaded from zenodo:
        # https://zenodo.org/record/8275347/files/audiotransformer_base_mAP_4999.pt?download=1
        # print('config encoder: ', config['encoder'])
        # print('config: ', config)

        # encoder
        self.encoder_type = config['encoder']
        self.teach = config['teach']
        self.temp = config['temp']
        self.contrastive_loss_type = config['contrastive_loss_type']
        self.alpha = config['alpha']

        if self.contrastive_loss_type == "infonce":
            self.contrastive_temp = nn.Parameter(
                    data=torch.tensor([2.995]), requires_grad=True
                )
            self.beta = None
        elif self.contrastive_loss_type == "siglip":
            self.contrastive_temp = nn.Parameter(
                    data=torch.tensor([2.995]), requires_grad=True
                )
            self.beta = nn.Parameter(data=torch.tensor([10.0]), requires_grad=True)
        else:
            self.contrastive_loss_type = None

        if config['encoder'] == "clap":        
            # print("Using CLAP encoder")    
            encoder_name = "laion/clap-htsat-fused"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name) 
            self.encoder = ClapAudioModel.from_pretrained(encoder_name)
            self.encoder.embed_dim = 768
        else:
            ced_config = CEDConfig()
            ced_checkpoint = torch.load(
                "pretrained_models/ced/audiotransformer_base_mAP_4999.pt"
            )
            self.encoder = AudioTransformer(
                ced_config,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                outputdim=527,
                target_length=1012,
            )
            self.encoder.load_state_dict(ced_checkpoint, strict=False)
        
            encoder_peft_config = LoraConfig(
                target_modules=["q_proj", "v_proj"],
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.apply_encoder_strategy(encoder_peft_config)
        
        self.fft_mlp = FFTAdapter(self.encoder.embed_dim)
        # mlp
        self.speech_former, self.speech_query_tokens = self.build_audio_qformer(
            1, self.encoder.embed_dim, 2, 1
        )

        # decoder
        hf_token = "your huggingface token"
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "llama_weight/LLaMA-7B-HF"
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.decoder = LlamaForCausalLM.from_pretrained(
            "llama_weight/LLaMA-7B-HF"
        )
        peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.apply_decoder_strategy(peft_config)

        # mlp, must call after init self.decoder
        self.enc_to_dec_proj = self.build_audio_projector(
            projector_type="linear", in_dim=self.speech_former.config.hidden_size
        )

    def print_module_parameters(self):
        encoder_num_params = sum([i.numel() for i in self.encoder.parameters()])
        decoder_num_params = sum([i.numel() for i in self.decoder.parameters()])
        speech_former_num_params = sum(
            [i.numel() for i in self.speech_former.parameters()]
        )
        mlp_num_params = sum([i.numel() for i in self.enc_to_dec_proj.parameters()])
        print(
            f"model params encoder: {encoder_num_params}, decoder: {decoder_num_params}, speech_former: {speech_former_num_params}, mlp: {mlp_num_params}"
        )

    def prepare_inputs_labels_for_multimodal(
        self, audio_embeds, atts, prompt, text=None, caps=None
    ):            
        prompt_left = []
        prompt_right = []
        prompt_left_no_rag = []
        prompt_right_no_rag = []


        if caps is not None and any(c is not None for c in caps):
            for i, (p, c) in enumerate(zip(prompt, caps)):
                l, r = p.split("<AcousticTokens>")
                l = "Similar audios sound like:\n\n" + c + "\n\n" + l
                # print('l: ',l)
                prompt_left.append(self.tokenizer.bos_token + l) 
                prompt_right.append(r)

            if str(self.teach).lower() == 'true':
                print('Teach:', self.temp)
                for i, (p, c) in enumerate(zip(prompt, caps)):
                    l, r = p.split("<AcousticTokens>")
                    prompt_left_no_rag.append(self.tokenizer.bos_token + l)
                    prompt_right_no_rag.append(r)
            
        else:
            self.teach = 'false'
            for i, p in enumerate(prompt):
                l, r = p.split("<AcousticTokens>")
                prompt_left.append(self.tokenizer.bos_token + l)
                prompt_right.append(r)

        prompt_left_tokens = self.tokenizer(
            prompt_left, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(audio_embeds.device)
        # print(prompt_left_tokens)
        prompt_left_embeds = self.decoder.model.model.embed_tokens(
            prompt_left_tokens.input_ids
        )

        prompt_right_tokens = self.tokenizer(
            prompt_right,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        ).to(audio_embeds.device)
        prompt_right_embeds = self.decoder.model.model.embed_tokens(
            prompt_right_tokens.input_ids
        )
        # print(prompt_left_tokens.input_ids)
        # print(prompt_left_tokens.input_ids.shape)
        input_embeds = torch.cat(
            [prompt_left_embeds, audio_embeds, prompt_right_embeds], dim=1
        )
        input_mask = torch.cat(
            [
                prompt_left_tokens.attention_mask,
                atts,
                prompt_right_tokens.attention_mask,
            ],
            dim=1,
        )

        if str(self.teach).lower() == 'true':
            prompt_left_tokens_no_rag = self.tokenizer(
                prompt_left_no_rag, add_special_tokens=False, padding=True, return_tensors="pt"
            ).to(audio_embeds.device)
            prompt_left_embeds_no_rag = self.decoder.model.model.embed_tokens(
                prompt_left_tokens_no_rag.input_ids
            )

            prompt_right_tokens_no_rag = self.tokenizer(
                prompt_right_no_rag,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            ).to(audio_embeds.device)
            prompt_right_embeds_no_rag = self.decoder.model.model.embed_tokens(
                prompt_right_tokens_no_rag.input_ids
            )

            input_embeds_no_rag = torch.cat(
                [prompt_left_embeds_no_rag, audio_embeds, prompt_right_embeds_no_rag], dim=1
            )
            input_mask_no_rag = torch.cat(
                [
                    prompt_left_tokens_no_rag.attention_mask,
                    atts,
                    prompt_right_tokens_no_rag.attention_mask,
                ],
                dim=1,
            )

        decoder_targets = None
        decoder_targets_no_rag = None
        contrastive_embeds = None
        contrastive_mask = None
        st, end, st_no_rag, end_no_rag = None, None, None, None
        if text is not None:
            new_text = []
            for t in text:
                new_text.append(t + self.tokenizer.eos_token)  # </s> is the eos_token
            text_tokens = self.tokenizer(
                new_text,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            ).to(audio_embeds.device)
            text_embeds = self.decoder.model.model.embed_tokens(text_tokens.input_ids)

            if self.contrastive_loss_type is not None:
                contrastive_embeds = text_embeds.clone()
                contrastive_mask = text_tokens.attention_mask.clone()

            targets = text_tokens.input_ids.masked_fill(
                text_tokens.input_ids == self.tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones([input_mask.shape[0], input_mask.shape[1]], dtype=torch.long)
                .to(audio_embeds.device)
                .fill_(-100)
            )
            decoder_targets = torch.cat([empty_targets, targets], dim=1)

            st = input_embeds.shape[1]
            input_embeds = torch.cat([input_embeds, text_embeds], dim=1)
            input_mask = torch.cat([input_mask, text_tokens.attention_mask], dim=1)
            end = input_embeds.shape[1]
            assert st != end

            if str(self.teach).lower() == 'true':
                empty_targets_no_rag = (
                    torch.ones([input_mask_no_rag.shape[0], input_mask_no_rag.shape[1]], dtype=torch.long)
                    .to(audio_embeds.device)
                    .fill_(-100)
                )
                decoder_targets_no_rag = torch.cat([empty_targets_no_rag, targets], dim=1)

                st_no_rag = input_embeds_no_rag.shape[1]
                input_embeds_no_rag = torch.cat([input_embeds_no_rag, text_embeds], dim=1)
                input_mask_no_rag = torch.cat([input_mask_no_rag, text_tokens.attention_mask], dim=1)
                end_no_rag = input_embeds_no_rag.shape[1]
                assert st_no_rag != end_no_rag
                assert (end_no_rag - st_no_rag) == (end - st)
        
        if str(self.teach).lower() == 'true':
            return input_embeds, input_mask, decoder_targets, input_embeds_no_rag, input_mask_no_rag, decoder_targets_no_rag, [st, end, st_no_rag, end_no_rag], contrastive_embeds, contrastive_mask
        
        # print("====================================")
        return input_embeds, input_mask, decoder_targets, contrastive_embeds, contrastive_mask

    def forward_encoder(self, audios):
        
        if self.encoder_type == "clap":
            audio_input = self.feature_extractor(audios, sampling_rate=48000, return_tensors="pt")
            audio_input['input_features'] = audio_input['input_features'].to(self.encoder.device)
            print(audio_input['input_features'].shape)
            encodings = self.encoder(input_features=audio_input['input_features'],is_longer=audio_input['is_longer'],output_hidden_states=True)
            encodings = torch.flatten(encodings.last_hidden_state,2)
            audio_embeds = encodings.permute(0,2,1)
            print('audio embeds:', audio_embeds.shape)
        else:
            audio_embeds = self.encoder(audios)


        # Qformer
        # import pdb
        # pdb.set_trace()
        #######################
        audio_embeds = self.fft_mlp(audio_embeds)
        #######################
        batch, tokens, dim = audio_embeds.shape
        kernel = (1, 17)  # for ced 714ms/per frame (ced 10s: 252 frame), we reduce to about 1.4 frames/second
        audio_embeds_new = F.unfold(
            audio_embeds.transpose(1, 2).unsqueeze(2), kernel_size=kernel, stride=kernel
        )
        audio_embeds_new = audio_embeds_new.view(batch, dim, kernel[1], -1)
        audio_embeds_new = torch.permute(audio_embeds_new, [0, 3, 2, 1])
        audio_embeds = audio_embeds_new.reshape(-1, kernel[1], dim)

        speech_atts = torch.ones(
            audio_embeds.size()[:-1], dtype=torch.long, device=audio_embeds.device
        )
        query_tokens = self.speech_query_tokens.expand(audio_embeds.shape[0], -1, -1)
        # print(query_tokens.shape)
        audio_embeds = self.speech_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )["last_hidden_state"]
        # print(audio_embeds.shape)
        # MLP
        encoder_hidden_states = self.enc_to_dec_proj(audio_embeds)
        encoder_hidden_states = encoder_hidden_states.view(
            batch, -1, encoder_hidden_states.size(2)
        ).contiguous()
        encoder_atts = torch.ones(
            encoder_hidden_states.size()[:-1], dtype=torch.long
        ).to(encoder_hidden_states.device)
        return encoder_hidden_states, encoder_atts
    
    def distillation_loss(self, student_logits, teacher_logits, temperature=1.0, alpha=0.5):
        """
        Computes the distillation loss between the student and teacher models.

        Args:
            student_logits (torch.Tensor): Logits from the student model, shape (batch_size, seq_len, dim)
            teacher_logits (torch.Tensor): Logits from the teacher model, shape (batch_size, seq_len, dim)
            temperature (float): Temperature for softening the probability distributions
            alpha (float): Weighting factor for distillation loss versus standard loss

        Returns:
            torch.Tensor: Computed distillation loss
        """
        # Apply temperature scaling
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        
        # Compute KL divergence manually: KL(student || teacher)
        kl_loss = torch.sum(student_probs * (torch.log(student_probs) - teacher_probs), dim=-1)
        kl_loss = kl_loss.mean()
        kl_loss *= (temperature ** 2)
        
        return kl_loss

    def forward(self, samples):
        audios = samples["audios"]
        text = samples["text"]
        retrieve_caps = samples["caps"]
        prompt = [random.choice(self.prompt)] * len(text)
        # encoder
        encoder_hidden_states, encoder_atts = self.forward_encoder(audios)

        
        if str(self.teach).lower() == 'true':
            input_embeds, input_mask, decoder_targets, input_embeds_no_rag, input_mask_no_rag, decoder_targets_no_rag, coord, contrastive_embeds, contrastive_mask = (
                self.prepare_inputs_labels_for_multimodal(
                    encoder_hidden_states, encoder_atts, prompt, text, retrieve_caps
                )
            )
            decoder_output_no_rag = self.decoder(
                input_ids=None,
                inputs_embeds=input_embeds_no_rag,
                attention_mask=input_mask_no_rag,
                labels=decoder_targets_no_rag,
                return_dict=True,
            )
            logits_no_rag = decoder_output_no_rag.logits[:, coord[2]:coord[3], :]
        else:
            input_embeds, input_mask, decoder_targets, contrastive_embeds, contrastive_mask = (
                self.prepare_inputs_labels_for_multimodal(
                    encoder_hidden_states, encoder_atts, prompt, text, retrieve_caps
                )
            )
        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
            labels=decoder_targets,
            return_dict=True,
        )
        logits = decoder_output.logits[:, coord[0]:coord[1], :].detach()
        distil_loss = self.distillation_loss(logits, logits_no_rag, temperature=self.temp)

        lossAlign = 0
        if self.contrastive_loss_type:
            # contrastive_out = self.decoder.model.model(
            #     input_ids=None,
            #     inputs_embeds=contrastive_embeds,
            #     attention_mask=contrastive_mask,
            #     return_dict=True,
            # )[0].detach()
            contrastive_out = contrastive_embeds.detach()
            assert len(contrastive_out.shape) == 3
            matrixSimi = average_similarity(encoder_hidden_states, contrastive_out, self.contrastive_temp)
            print(matrixSimi)
            lossAlign = compute_loss(self.contrastive_loss_type, matrixSimi, self.beta)
            lossAlign = self.alpha*lossAlign

        if str(self.teach).lower() == 'true':
            print(f"=======temperature: {self.temp}, Loss teacher: {decoder_output.loss.item()}, Loss student: {decoder_output_no_rag.loss}, Loss distil: {distil_loss}, loss contrastive: {lossAlign}")
            return (decoder_output.loss + decoder_output_no_rag.loss + distil_loss + lossAlign), decoder_output.logits, decoder_output.loss, decoder_output_no_rag.loss, distil_loss, lossAlign
        else:
            return decoder_output.loss + lossAlign, decoder_output.logits

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=2,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        audios = samples["audios"].to(self.device)
        retrieve_caps = samples["caps"]

        prompt = [random.choice(self.prompt)] * audios.shape[0]
        encoder_hidden_states, encoder_atts = self.forward_encoder(audios)
        if (str(self.teach).lower() == 'true') and (retrieve_caps is not None and any(c is not None for c in retrieve_caps)):
            input_embeds, input_mask, decoder_targets, input_embeds_no_rag, input_mask_no_rag, decoder_targets_no_rag, coord, contrastive_embeds, contrastive_mask = (
                self.prepare_inputs_labels_for_multimodal(
                    encoder_hidden_states, encoder_atts, prompt, caps=retrieve_caps
                )
            )
        else:
            input_embeds, input_mask, decoder_targets, contrastive_embeds, contrastive_mask = (
                self.prepare_inputs_labels_for_multimodal(
                    encoder_hidden_states, encoder_atts, prompt, caps=retrieve_caps
                )
            )

        outputs = self.decoder.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=1.0,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        captions = self.tokenizer.batch_decode(outputs, add_special_tokens=False)
        return captions
