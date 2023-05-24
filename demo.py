import gradio as gr
import torch
from PIL import Image
import numpy as np
from transformers import ViTImageProcessor
from models.swinllama import SwinLLama
from configs.config_mimic import parser


class SwinLLamaR2gen:
    def __init__(self, device):
        print("Initializing SwinLLama to %s" % device)
        self.device = device
        self.image_processor = ViTImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.model = SwinLLama(args)
        self.model.to(device)
        self.model.eval()
       
        
    def inference(self, image_path, max_output_tokens):
        try:
            array = np.array(image_path)
            pixel_values = self.image_processor(array, return_tensors="pt").pixel_values
            image = pixel_values.to(self.device)
            
            img_embeds, atts_img = self.model.encode_img(image)
            img_embeds = self.model.layer_norm(img_embeds)
            
            img_embeds, atts_img = self.model.prompt_wrap(img_embeds, atts_img)
            
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=atts_img.dtype,
                            device=atts_img.device) * self.model.llama_tokenizer.bos_token_id
            bos_embeds = self.model.embed_tokens(bos)
            inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
                 
            output_token = self.model.llama_model.generate(
                inputs_embeds=inputs_embeds,
                no_repeat_ngram_size=2,
                num_beams=3,
                do_sample=False,
                max_length=max_output_tokens,
                repetition_penalty=1,
                length_penalty=1,
                temperature=0,
            )
            output_token = output_token[0]
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Radiologist:')[-1].strip()
            output_text = output_text.replace('impression', 'Impression').replace('findings', '\n\nFindings')
            return output_text
        except Exception as e:
            return e


if __name__ == "__main__":
    
    args = parser.parse_args()
    model = SwinLLamaR2gen('cuda:0')
    imagebox = gr.Image(interactive=True).style(height=270)
    with gr.Accordion("Parameters", open=True, visible=True) as parameter_row:
        max_output_tokens = gr.Slider(minimum=0, maximum=160, value=80, step=20, interactive=True, label="Max output tokens",)
        
    demo = gr.Interface(model.inference, inputs=[imagebox, max_output_tokens], outputs=gr.TextArea(label="SwinLlama").style(show_copy_button=True))
    demo.queue(concurrency_count=3).launch()