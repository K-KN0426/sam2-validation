# assets

| ファイル | 内容 | 作成方法 |
| --- | --- | --- |
| `sample.png` | 図形の画像（640×480） | `scripts/make_sample_image.py` で生成 |
| `cell_sample.png` | 細胞の画像（704×512） | Google Gemini で生成した画像の一部を切り出し |
| `street_sample.webp` | 街並みと人物の画像（1408×768） | 画像生成 AI で生成 |

## AI 生成画像（cell_sample.png, street_sample.webp）

- AI 生成画像。実在の人物・患者・検体・場所を撮影したものではない。
- Gemini の生成画像には [SynthID](https://deepmind.google/technologies/synthid/) の電子透かしが含まれる場合がある。
- 利用は各生成サービスの利用規約に従う（Google Gemini の場合は[生成 AI の使用禁止に関するポリシー](https://policies.google.com/terms/generative-ai/use-policy)を含む）。
- このリポジトリでは本画像の権利を主張せず、現状のまま提供する。
- `outputs/cell_sam2_*.png` は `cell_sample.png` に推論結果を重ねたもので、扱いは同じ。
