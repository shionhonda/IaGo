# What is IaGo?
IaGo is an Othello AI using SL(supervised learning) policy network inspired by AlphaGo.  
It doesn't predict plays ahead, but it is still powerful.  

# How to play?  
1. Install chainer  
`$ pip install chainer`  

2. Download this repository  
`$ git clone git@github.com:shionhonda/IaGo.git`  

3. Move to IaGo directory  

4. Execute game.py  
`$ python game.py`

5. When placing a stone, input two numbers seperated by comma. For example:  
`4,3`  
The first number corresponds to the vertical position and the second to the horizontal.

# How to learn?
1. Download data from <http://meipuru-344.hatenablog.com/entry/2017/11/27/205448>  
2. Save it as IaGo/data/data.txt    
3. Execute train.py  
`$ python train.py`  
You need GPU to complete this step. It will take about 3 hours.

# Acknowledgements  
Special thanks to @lazmond3 for giving lots of feedbacks!
