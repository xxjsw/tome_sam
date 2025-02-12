from evaluate import EvaluateArgs, evaluate
from tome_sam.utils.tome_presets import SAMToMeSetting, ToMeSD, ToMeConfig, PiToMe, ToMe
from flops import get_flops

tomesd_setting: SAMToMeSetting = {
    7: ToMeConfig(
        mode='tomesd',
        params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
    ),
    8: ToMeConfig(
        mode='tomesd',
        params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
    ),
    9: ToMeConfig(
        mode='tomesd',
        params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
    ),
    10: ToMeConfig(
        mode='tomesd',
        params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
    ),
    11: ToMeConfig(
        mode='tomesd',
        params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=False)
    ),
}

tome_setting: SAMToMeSetting = {
    2: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
    5: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
    8: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
    11: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
}

pitome_setting: SAMToMeSetting = {
    7: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
    8: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
    9: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
    10: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
    11: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
}


if __name__ == '__main__':
    test_cases = [None, tome_setting, pitome_setting, tomesd_setting]
    for setting in test_cases:
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
            tome_setting=setting,
        )
        eval_results = evaluate(evaluate_args)
        print(eval_results)

        flops_per_image = get_flops(evaluate_args)
        print(flops_per_image)
