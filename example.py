from evaluate import EvaluateArgs, evaluate
from tome_sam.utils.tome_presets import SAMToMeSetting, ViTToMeConfig, BSMToMe, ToMeConfig, PiToMe
from flops import get_flops

if __name__ == '__main__':
    test_bsm_setting: SAMToMeSetting = {
        2: ViTToMeConfig(
            kv=ToMeConfig(
                mode='bsm',
                params=BSMToMe(r=0.5, sx=2, sy=2)
            ),
            q=ToMeConfig(
                mode='bsm',
                params=BSMToMe(r=0.5, sx=2, sy=2)
            )
        ),
        5: ViTToMeConfig(
            kv=ToMeConfig(
                mode='bsm',
                params=BSMToMe(r=0.5, sx=2, sy=2)
            ),
            q=ToMeConfig(
                mode='bsm',
                params=BSMToMe(r=0.5, sx=2, sy=2)
            )
        ),
        8: ViTToMeConfig(
            kv=ToMeConfig(
                mode='bsm',
                params=BSMToMe(r=0.5, sx=2, sy=2)
            ),
            q=ToMeConfig(
                mode='bsm',
                params=BSMToMe(r=0.5, sx=2, sy=2)
            )
        ),
        11: ViTToMeConfig(
            kv=ToMeConfig(
                mode='bsm',
                params=BSMToMe(r=0.5, sx=2, sy=2)
            ),
            q=ToMeConfig(
                mode='bsm',
                params=BSMToMe(r=0.5, sx=2, sy=2)
            )
        ),
    }

    test_pitome_setting = SAMToMeSetting = {
        2: ViTToMeConfig(
            kv=ToMeConfig(
                mode='pitome',
                params=PiToMe(r=0.5, margin=0.9, alpha=1.0)
            ),
            q=ToMeConfig(
                mode='pitome',
                params=PiToMe(r=0.5, margin=0.9, alpha=1.0)
            )
        ),
        5: ViTToMeConfig(
            kv=ToMeConfig(
                mode='pitome',
                params=PiToMe(r=0.5, margin=0.9, alpha=1.0)
            ),
            q=ToMeConfig(
                mode='pitome',
                params=PiToMe(r=0.5, margin=0.9, alpha=1.0)
            )
        ),
        8: ViTToMeConfig(
            kv=ToMeConfig(
                mode='pitome',
                params=PiToMe(r=0.5, margin=0.9, alpha=1.0)
            ),
            q=ToMeConfig(
                mode='pitome',
                params=PiToMe(r=0.5, margin=0.9, alpha=1.0)
            )
        ),
        11: ViTToMeConfig(
            kv=ToMeConfig(
                mode='pitome',
                params=PiToMe(r=0.5, margin=0.9, alpha=1.0)
            ),
            q=ToMeConfig(
                mode='pitome',
                params=PiToMe(r=0.5, margin=0.9, alpha=1.0)
            )
        ),
    }

    evaluate_args = EvaluateArgs(
        dataset="dis",
        output="",
        model_type="vit_b",
        checkpoint="checkpoints/sam_vit_b_01ec64.pth",
        device="mps",
        seed=0,
        input_size=[1024, 1024],
        batch_size=1,
        multiple_masks=False,
        tome_setting = None,
    )

    iou_results = evaluate(evaluate_args)
    print(iou_results)

    # flops_per_image = get_flops(evaluate_args)
    # print(flops_per_image)
