# 遊び方  
1. chainerをインストール  
`$ pip install chainer`  

2. 本リポジトリを丸ごとダウンロード  
`$ git clone git@github.com:shionhonda/DeepOthello.git`  

3. DeepOthelloディレクトリに移動  

4. game.pyを実行  
`$ python game.py`

5. 石を置くときはカンマ区切りで1-8の数字を2つ入力してください。例えば  
`4,3`  
のように。縦軸が先、横軸が後です。

# 学習させたいとき
<http://meipuru-344.hatenablog.com/entry/2017/11/27/205448>のデータをダウンロードして、  
DeepOthello/data/data.txt  
として保存し、  
`$ python train.py`  
を実行してください。今のところGPUが無いと計算できません。3時間くらいで終わると思います。
