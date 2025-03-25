from segment_anything import sam_model_registry
import torch

def get_model(model_type: str, checkpoint: str):
    return sam_model_registry[model_type](
            checkpoint=checkpoint)


if __name__ == '__main__':
    model_types = ["vit_b", "vit_l", "vit_h"]
    checkpoints = ['sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth', 'sam_vit_h_4b8939.pth']

    for model_type, checkpoint in zip(model_types, checkpoints):
        sam = get_model(model_type, 'checkpoints/'+checkpoint)
        total_params = sum(p.numel() for p in sam.parameters())
        image_encoder = sam.image_encoder
        image_encoder_params = sum(p.numel() for p in image_encoder.parameters())
        prompt_encoder = sam.prompt_encoder
        prompt_encoder_params = sum(p.numel() for p in prompt_encoder.parameters())
        mask_decoder = sam.mask_decoder
        mask_decoder_params = sum(p.numel() for p in mask_decoder.parameters())
        print(f'SAM model type: {model_type}')
        print(f'Total params: {total_params}')
        print(f'Image encoder params: {image_encoder_params}')
        print(f'Prompt encoder params: {prompt_encoder_params}')
        print(f'Image decoder params: {mask_decoder_params}')

