日本語は後半にあります。  

# What is DeepOthello?
DeepOthello is an Othello AI using SL(supervised learning) policy network inspired by AlphaGo.  
It doesn't predict plays ahead, but it is still powerful.  

# How to play  
1. Install chainer  
`$ pip install chainer`  

2. Download this repository  
`$ git clone git@github.com:shionhonda/DeepOthello.git`  

3. Move to DeepOthello directory  

4. Execute game.py  
`$ python game.py`

5. When placing a stone, input two numbers seperated by comma. For example:  
`4,3`  
The first number corresponds to the vertical position and the second to the horizontal.

# How to learn
1. Download data from <http://meipuru-344.hatenablog.com/entry/2017/11/27/205448>  
2. Save it as DeepOthello/data/data.txt    
3. Execute train.py  
`$ python train.py`  
You need GPU to complete this step. It will take about 3 hours. 

# DeepOthelloとは  
アルファ碁で使われているSLポリシーネットワークを利用したオセロAIです。  
先読みはせず、現在の局面からその都度、最適な手を選んでいるだけですが、それでも強いです。  
作成者の自分でも勝てません。  

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
