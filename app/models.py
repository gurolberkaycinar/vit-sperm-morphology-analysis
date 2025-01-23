from transformers import BeitForImageClassification, DeiTForImageClassification, Swinv2ForImageClassification, ConvNextForImageClassification, ConvNextV2ForImageClassification, Dinov2ForImageClassification

def get_model(model_name):
    match model_name:
        case "convnextv2tiny":
            return ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-tiny-22k-224",
        num_labels=18,
        ignore_mismatched_sizes=True
    )
        case "deitbase":
            return DeiTForImageClassification.from_pretrained(
        "facebook/deit-base-distilled-patch16-224",
        num_labels=18
    )
        case "beitbase":
            return BeitForImageClassification.from_pretrained(
        "microsoft/beit-base-patch16-224-pt22k",
        num_labels=18
    )
        case "dinov2small":
            return Dinov2ForImageClassification.from_pretrained(
        "facebook/dinov2-small",
        num_labels=18,
        ignore_mismatched_sizes=True
    )
        case "swinv2tiny":
            return Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256",
        num_labels=18,
        ignore_mismatched_sizes=True
    )       
        case "convnextv2tiny":
            return ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-tiny-22k-224",
        num_labels=18,
        ignore_mismatched_sizes=True
    )
        case "convnextv2large":
            return ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-large-22k-224",
        num_labels=18,
        ignore_mismatched_sizes=True
    )
        case default:
            return

