Input Data
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    INPUT EMBEDDING LAYER                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Gate Embedding │  │Position Embedding│  │Parameter Emb │ │
│  │   [vocab_size,  │  │   [max_pos,     │  │  [1, d_model]│ │
│  │    d_model]     │  │    d_model]     │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                              ↓                              │
│                    Element-wise Addition                    │
│                              ↓                              │
│                        Dropout(0.1)                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                 CONSTRAINT ENCODER BLOCKS                   │
│                      (n_layers // 2)                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                TRANSFORMER BLOCK                        │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │              PRE-NORM STRUCTURE                     │ │ │
│  │  │                                                     │ │ │
│  │  │  Input x ──┐                                        │ │ │
│  │  │            │                                        │ │ │
│  │  │            ↓                                        │ │ │
│  │  │      LayerNorm(eps=1e-6)                           │ │ │
│  │  │            ↓                                        │ │ │
│  │  │    ModularAttention(advanced)                      │ │ │
│  │  │    ├─ GridPositionalAttention                      │ │ │
│  │  │    ├─ RegisterFlowAttention                        │ │ │
│  │  │    ├─ EntanglementAttention                        │ │ │
│  │  │    ├─ SemanticAttention                            │ │ │
│  │  │    └─ AttentionFusionNetwork                       │ │ │
│  │  │            ↓                                        │ │ │
│  │  │      Dropout(dropout)                              │ │ │
│  │  │            ↓                                        │ │ │
│  │  │    Scale * attention_out                           │ │ │
│  │  │            ↓                                        │ │ │
│  │  │            ├─────────────────┐ (Skip Connection 1)  │ │ │
│  │  │            ↓                 │                     │ │ │
│  │  │         Add(x)  ←────────────┘                     │ │ │
│  │  │            ↓                                        │ │ │
│  │  │      LayerNorm(eps=1e-6)                           │ │ │
│  │  │            ↓                                        │ │ │
│  │  │      Feed Forward Network                          │ │ │
│  │  │      ├─ Linear(d_model → d_ff)                     │ │ │
│  │  │      ├─ GELU()                                     │ │ │
│  │  │      ├─ Dropout(dropout)                           │ │ │
│  │  │      ├─ Linear(d_ff → d_model)                     │ │ │
│  │  │      └─ Dropout(dropout)                           │ │ │
│  │  │            ↓                                        │ │ │
│  │  │    Scale * ff_out                                  │ │ │
│  │  │            ↓                                        │ │ │
│  │  │            ├─────────────────┐ (Skip Connection 2)  │ │ │
│  │  │            ↓                 │                     │ │ │
│  │  │      Add(previous_x) ←───────┘                     │ │ │
│  │  │            ↓                                        │ │ │
│  │  │        Output x                                     │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│               SEQUENCE DECODER BLOCKS                       │
│                  (n_layers - n_layers//2)                  │
│  [Same TransformerBlock structure as above]                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    ACTION HEADS                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   GATE HEAD     │  │  POSITION HEAD  │  │PARAMETER HEAD│ │
│  │                 │  │                 │  │              │ │
│  │ Linear(d_model  │  │ Linear(d_model  │  │Linear(d_model│ │
│  │   → d_model/2)  │  │   → d_model/2)  │  │  → d_model/4)│ │
│  │       ↓         │  │       ↓         │  │      ↓       │ │
│  │    GELU()       │  │    GELU()       │  │   GELU()     │ │
│  │       ↓         │  │       ↓         │  │      ↓       │ │
│  │ Linear(d_model/2│  │ Linear(d_model/2│  │Linear(d_model│ │
│  │ → n_gate_types) │  │ → position_dim) │  │   /4 → 1)    │ │
│  │                 │  │                 │  │              │ │
│  │  Classification │  │  Classification │  │  Regression  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘