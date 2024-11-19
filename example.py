from evaluate import EvaluateArgs, evaluate
from tome_sam.utils.tome_presets import SAMToMeSetting, ViTToMe, BSMToMe


test_tome_setting: SAMToMeSetting = {
    0: ViTToMe(
        kv_mode=BSMToMe(r=0.6, sx=2, sy=2),
        q_mode=BSMToMe(r=0.8, sx=4, sy=4),
    ),
    3: ViTToMe(
        kv_mode=BSMToMe(r=0.5, sx=3, sy=3),
        q_mode=BSMToMe(r=0.7, sx=5, sy=5),
    ),
    5: ViTToMe(
        kv_mode=BSMToMe(r=0.4, sx=1, sy=1),
        q_mode=BSMToMe(r=0.9, sx=6, sy=6),
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

results = evaluate(evaluate_args)
print(results)
