# What is IaGo?
IaGo is an Othello AI using SL(supervised learning) policy network, value network, and MCTS(Monte Carlo tree search) inspired by AlphaGo.  
Short description in English:  
[IaGo: an Othello AI inspired by AlphaGo](https://www.slideshare.net/ShionHonda/iago-an-othello-ai-inspired-by-alphago)  
Description in Japanese:  
[AlphaGoを模したオセロAIを作る(1): SLポリシーネットワーク - Qiita](https://qiita.com/shionhonda/items/7a3eb79f55299e743630)  
[AlphaGoを模したオセロAIを作る(2): RLポリシーネットワーク - Qiita](https://qiita.com/shionhonda/items/56e37872419a3c79b3aa)  
[AlphaGoを模したオセロAIを作る(3): バリューネットワーク - Qiita](https://qiita.com/shionhonda/items/7dce679b385f738a0dcb)  

# How to play?  
1. Install chainer  
`$ pip install chainer`  

2. Download this repository  
`$ git clone git@github.com:shionhonda/IaGo.git`  

3. Execute game.py  
`$ python ./IaGo/game.py`  
You can set following options:  
`--auto=False` or `--a=False`  
If this is set True, autoplay begins between SLPolicy and APV-MCTS, and if False (default), the game is played by you and APV-MCTS.  
`--level=6`  
This option is the maximum depth of tree search for APV-MCTS. The higher level means deeper search. Note that there is a trade-off between level (or strength, search depth) and computational complexity (thinking time). The thinking time is exponential to the level. The default value is 6.  
4. When placing a stone, input two numbers seperated by comma. For example:  
`4,3`  
The first number corresponds to the vertical position and the second to the horizontal (one origin).

# How to train?
1. Download data from <http://meipuru-344.hatenablog.com/entry/2017/11/27/205448>
2. Save it as "IaGo/data/data.txt"    
3. Augment data  
`$ python load.py`  
You need at least 32MB RAM to complete this step.  
4. Execute train_policy.py to train SL policy network.  
`$ python train_policy.py --policy=sl --epoch=30 --gpu=0`  
You need GPUs to complete this step. It will take about 2 days.
5. Execute train_policy.py to train rollout policy.  
`$ python train_policy.py --policy=rollout --epoch=1 --gpu=0`  
This is fast.  
6. Execute train_rl.py to reinforce SL policy network with REINFORCE (a kind of policy gradients).  
`$ python train_policy.py --set=10000`
7. Execute train_value.py to train value network.  
`$ python train_value.py --epoch=20 --gpu=0`  
8. Training done!

# Acknowledgements  
Special thanks to:  
@Rochestar-NRT for replication of AlphaGo (especially MCTS).  
@lazmond3 for giving lots of feedbacks!
