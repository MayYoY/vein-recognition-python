from torchvision import transforms, models
from enhancement import gabor, hist_method, retinex, stretch


class NeuralFinger:
    net = models.resnet18()
    train_path = ""
    test_path = ""

    enhance = None
    train_trans = transforms.Compose([transforms.ToTensor(),
                                      transforms.ColorJitter(brightness=0.05, contrast=0.05,
                                                             saturation=0.05, hue=0.05),
                                      transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                                              scale=(0.95, 1.05)),
                                      transforms.Resize((224, 224))])
    test_trans = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224, 224))])
    batch_size = 24
    num_epochs = 30
    lr = 0.0005  # baseline 0.5, resnet 0.0005
    weight_decay = 0.0
    device = "cuda"
    save = True
    name = f"resnet18_finger_light1_epoch{num_epochs}_batch{batch_size}" \
           f"_lr{lr}_weightdecay{weight_decay}"


class NeuralPalm:
    c = 1
    train_path = f""
    test_path = f""

    enhance = None
    train_trans = transforms.Compose([transforms.ToTensor(),
                                      transforms.ColorJitter(brightness=0.05, contrast=0.05,
                                                             saturation=0.05, hue=0.05),
                                      transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                                              scale=(0.95, 1.05)),
                                      transforms.Resize((224, 224))])
    test_trans = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224, 224))])
    batch_size = 32
    num_epochs = 30
    lr = 0.08  # baseline 0.08, resnet 0.0008
    weight_decay = 0.0
    device = "cuda"
    save = False
    name = ""


class NeuralInfer:
    test_trans = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224, 224))])
    enhance = None
    test_path = ""
    model_path = ""

    feature = True
    probability = True
    compare = True


class LBPFinger:
    train_path = ""
    test_path = ""

    enhance = None
    train = True  # 是否训练
    save = True
    name = ""


class LBPPalm:
    train_path = ""
    test_path = ""

    enhance = None
    train = True  # 是否训练
    save = True
    name = ""


class LBPShow:
    test_path = ""
    model_path = ""
    enhance = None


class TextureFinger:
    train_path = ""
    test_path = ""

    enhance = None
    train = True  # 是否训练
    save = True
    method = "repeatedLineTrack"
    name = ""


class TextureShow:
    test_path = ""
    model_path = ""

    enhance = None
    # method = "repeatedLineTrack"
    method = "maxCurvature"


class TexturePalm:
    train_path = ""
    test_path = ""

    enhance = None
    train = True  # 是否训练
    save = True
    method = "repeatedLineTrack"
    name = ""
