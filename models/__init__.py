from .baseline_fusion_gate import (
    ScaleCentricFusionWeightedFactor, ScaleCentricFusionWeightedMHA,
    ScaleCentricFusionWeightedSoftmax, ScaleCentricFusionWeightedRelationMatrix,
    ScaleCentricFusionGateFactor, ScaleCentricFusionGateMHA,
    ScaleCentricFusionGateSoftmax, ScaleCentricFusionGateRelationMatrix,
    ScaleCentricFusionFiLMFactor, ScaleCentricFusionFiLMMHA,
    ScaleCentricFusionFiLMSoftmax, ScaleCentricFusionFiLMRelationMatrix,
)
from .baseline_fusion_gate_viz import ScaleCentricFusionGateMHAViz
from .baseline_master import MASTERClassifier
from .loss import FocalLoss, SupConLoss
from .tslib_models.wrappers import (
    LSTMClassifier, GRUClassifier, TCNClassifier,
    TransformerClassifier, InformerClassifier, AutoformerClassifier,
    PatchTSTClassifier, TimesNetClassifier, iTransformerClassifier,
    TimeMixerClassifier, ModernTCNClassifier,
)
from .baseline_msmamba import MSMambaClassifier
from .baseline_rcib import RevINCNNiTransformerBiLSTMClassifier


def get_model(model_type='scale_centric_fusion_gate_mha', input_dim=9,
              hidden_dim=128, num_layers=2, num_classes=5, dropout_rate=0.3,
              kernel_size=3, num_levels=4, d_model=64, depth=4,
              encoder_type='gru', intra_hidden=32, lags=None,
              patch_size=5, patch_embed_dim=32, macro_dim=14, **kwargs_extra):
    kwargs = dict(
        per_scale_dim=hidden_dim, hidden_dim=hidden_dim,
        num_classes=num_classes, dropout_rate=dropout_rate,
        num_layers=num_layers, num_levels=num_levels,
        encoder_type=encoder_type, kernel_size=kernel_size,
        intra_hidden=intra_hidden, macro_dim=macro_dim,
    )

    # ---- 实验 L2: Gate 融合 × 4 种市场编码器（主实验） ----
    if model_type == 'scale_centric_fusion_gate_factor':
        return ScaleCentricFusionGateFactor(**kwargs)
    elif model_type == 'scale_centric_fusion_gate_mha':
        return ScaleCentricFusionGateMHA(**kwargs)
    elif model_type == 'scale_centric_fusion_gate_softmax':
        return ScaleCentricFusionGateSoftmax(**kwargs)
    elif model_type == 'scale_centric_fusion_gate_relation_matrix':
        return ScaleCentricFusionGateRelationMatrix(**kwargs)

    # ---- 可视化专用副本：与 fusion_gate_mha 权重完全一致 ----
    elif model_type == 'scale_centric_fusion_gate_mha_viz':
        return ScaleCentricFusionGateMHAViz(**kwargs)

    # ---- 实验 L1: Weighted 融合 × 4 种市场编码器 ----
    elif model_type == 'scale_centric_fusion_weighted_factor':
        return ScaleCentricFusionWeightedFactor(**kwargs)
    elif model_type == 'scale_centric_fusion_weighted_mha':
        return ScaleCentricFusionWeightedMHA(**kwargs)
    elif model_type == 'scale_centric_fusion_weighted_softmax':
        return ScaleCentricFusionWeightedSoftmax(**kwargs)
    elif model_type == 'scale_centric_fusion_weighted_relation_matrix':
        return ScaleCentricFusionWeightedRelationMatrix(**kwargs)

    # ---- 实验 L3: FiLM 融合 × 4 种市场编码器 ----
    elif model_type == 'scale_centric_fusion_film_factor':
        return ScaleCentricFusionFiLMFactor(**kwargs)
    elif model_type == 'scale_centric_fusion_film_mha':
        return ScaleCentricFusionFiLMMHA(**kwargs)
    elif model_type == 'scale_centric_fusion_film_softmax':
        return ScaleCentricFusionFiLMSoftmax(**kwargs)
    elif model_type == 'scale_centric_fusion_film_relation_matrix':
        return ScaleCentricFusionFiLMRelationMatrix(**kwargs)

    # ---- 对比实验：MASTER 市场引导 Transformer（需要 macro） ----
    elif model_type == 'baseline_master':
        return MASTERClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth, macro_dim=macro_dim,
        )

    # ---- 对比实验：无市场信息时序模型（统一分类头，来自 tslib_models） ----
    elif model_type == 'baseline_lstm':
        return LSTMClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
        )
    elif model_type == 'baseline_gru':
        return GRUClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
        )
    elif model_type == 'baseline_tcn':
        return TCNClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            num_levels=num_levels, kernel_size=kernel_size,
        )
    elif model_type == 'baseline_transformer':
        return TransformerClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth,
        )
    elif model_type == 'baseline_informer':
        return InformerClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth,
        )
    elif model_type == 'baseline_autoformer':
        return AutoformerClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth,
        )
    elif model_type == 'baseline_patchtst':
        return PatchTSTClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth, patch_size=patch_size,
        )
    elif model_type == 'baseline_timesnet':
        return TimesNetClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth,
        )
    elif model_type == 'baseline_itransformer':
        return iTransformerClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth,
        )
    elif model_type == 'baseline_timemixer':
        return TimeMixerClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth,
        )
    elif model_type == 'baseline_moderntcn':
        return ModernTCNClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth, kernel_size=kernel_size,
        )

            # ---- 对比实验：ms-Mamba 多尺度 Mamba + 统一分类头 ----
    elif model_type == 'baseline_msmamba':
        return MSMambaClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth,
            num_scales=kwargs_extra.get('num_scales', 4),
            d_state=kwargs_extra.get('d_state', 16),
            base_dt=kwargs_extra.get('base_dt', 0.01),
        )

    # ---- 对比实验：RevIN-CNN-iTransformer-BiLSTM 模块化框架 + 统一分类头 ----
    elif model_type == 'baseline_rcib':
        return RevINCNNiTransformerBiLSTMClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_classes=num_classes,
            dropout_rate=dropout_rate, intra_hidden=intra_hidden,
            d_model=d_model, depth=depth,
            cnn_channels=kwargs_extra.get('cnn_channels', 64),
            bilstm_hidden=kwargs_extra.get('bilstm_hidden', 128),
        )


    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. 本子项目支持的模型："
            f"scale_centric_fusion_gate_{{factor,mha,softmax,relation_matrix}}, "
            f"scale_centric_fusion_gate_mha_viz, "
            f"scale_centric_fusion_weighted_{{...}}, scale_centric_fusion_film_{{...}}, "
            f"baseline_master, "
            f"baseline_{{lstm,gru,tcn,transformer,informer,autoformer,patchtst,timesnet,"
            f"itransformer,timemixer,moderntcn}}。"
        )
