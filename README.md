# Photo Fail Russian Roulette

誰の事故画が撮れるかはお楽しみ！ランダムに指定された回数まばたきすると、自動でシャッターが切られます。

## 機能
- リアルタイムまばたき検出
- ランダムなまばたき回数の目標設定
- 目標達成時の自動撮影
- 顔のランドマーク表示オプション
- Eye Aspect Ratio (EAR) 表示オプション
- まばたきカウント表示オプション

## セットアップ手順

1. リポジトリのクローン:
```bash
git clone https://github.com/machitaro/Photo_Fail_Russian_Roulette.git
cd blink-detection
```

2. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```

3. 環境変数の設定:
```bash
# .env.exampleをコピーして.envを作成
cp .env.example .env
# 必要に応じて.envファイルを編集
```

4. 顔認識モデルのダウンロード:
```bash
# モデルファイルをダウンロード
chmod +x download_model.sh
./download_model.sh
```

5. アプリケーションの実行:
```bash
python app.py
```

6. ブラウザでアクセス:
```
http://localhost:5001
```

## 環境変数の設定

環境変数は開発環境と本番環境で異なる設定を使い分けるために使用します。
ご自身の開発環境に合わせて値を変更してください

## 検出パラメータの詳細

### Eye Aspect Ratio (EAR)しきい値
- **説明**: まばたりを検出するための目の開き具合の閾値
- **デフォルト値**: 0.2
- **影響**: 
  - 値が小さい: まばたき検出が厳密になる（完全に目を閉じる必要がある）
  - 値が大きい: まばたき検出が緩くなる（少し目を細めるだけで検出）

### 最小/最大まばたき回数
- **最小まばたき回数**:
  - **デフォルト値**: 1
  - **用途**: ランダムに設定される目標まばたき回数の下限

- **最大まばたき回数**:
  - **デフォルト値**: 5
  - **用途**: ランダムに設定される目標まばたき回数の上限
  
※ 最小値は最大値より小さい必要があります

### 表示設定オプション
1. **顔のランドマーク表示**
   - 顔の特徴点（68点）を画面上に表示
   - 位置確認用

2. **EAR値の表示**
   - 左右の目のEAR値をリアルタイム表示
   - 緑色: 目が開いている状態
   - 赤色: まばたりとして検出される状態

3. **まばたきカウント表示**
   - 現在のまばたき回数を表示

## 謝辞
- dlibの顔認識モデル: http://dlib.net/
- OpenCV: https://opencv.org/