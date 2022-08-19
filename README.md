# Contrastive-Learning-for-SS

これまでのルールベースのシーン分離手法はフレームの特徴量抽出にImageNetで事前学習したVGG16を使用していた。\
しかしImageNetは離散的な画像のデータセットであり、今回のタスクで必要となる時系列情報を考慮していない。\
そのためImageNetのVGG16による埋め込みは、無関係なデータ同士が中央の比較的近くに集合してしまう状況を作り出していた。\
この状況は閾値によるシーン分離の性能低下をもたらす。\
したがって今回はこの画像の特徴量の抽出部分において、時系列関係を考慮した学習手法を提案し、性能改善を試みる。\
ImageNet事前学習VGG16によるベクトルをt-SNEで次元削減したもの\
![ダウンロード (1)wetgeg](https://user-images.githubusercontent.com/55880071/185693909-de696ed7-fb00-42ae-b82e-4cb6eb4c1915.png)

提案手法によって抽出されたベクトルをt-SNEで次元削減したもの
![対象学習後](https://user-images.githubusercontent.com/55880071/185694363-158e05a0-c010-41bc-af35-f3e8ef6542b7.png)
