from .patchtst import Model as PatchTST
from .tft import Model as TFT
from .adapatch import Model as AdaPatch
from .gcformer import Model as GCformer
from .itransformer import Model as iTransformer
from .vanilla_transformer import Model as VanillaTransformer
from .timesnet import Model as TimesNet
from .dlinear import Model as DLinear

model_dict = {
    "PatchTST": PatchTST,
    "TFT": TFT,
    "AdaPatch": AdaPatch,
    "GCFormer": GCformer,
    "iTransformer": iTransformer,
    "VanillaTransformer": VanillaTransformer,
    "TimesNet": TimesNet,
    "DLinear": DLinear
}
