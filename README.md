# Semantic Textual Similarity Classification

## I. 문제 형태 (Quora Question Pair와 같은 형식) 
1. 두 문장이 주어진다
2. 같은 뜻인가? Binary Classification

### 제약사항 1: 실제 문장이 무엇인지는 모른다 
- 장점: cleaned-up, tokenized 되었으므로 tokenizer 선택 등등에서 할 일이 줄었다고 볼 수 있다
- 단점: vocab이 없다 -> pretrained word vector들은 사용할 수 없다. 마찬가지로 pretrained weight들도 사용하기 어려워 보인다
- 물론 multilingual NMT 시스템을 봤을 때 다른 vocabulary에서 학습된 걸 가져와서 사용해볼 수도 있었지만... 잘 된다고 해도 납득되는 방법은 아니라서 pass 그리고 vocab이랑 model이 너무 커진다.

### 제약사항 2: 데이터셋 사이즈가 작다 (40k instances)
- 유사 k-fold CV를 한다: 1:9 사이즈의 split을 세 개 만든다 
- 모델이 문장 1,2에 대해 symmetric하지 않다면 trainingset은 문장 1,2를 서로 바꾸어 augmentation할 수 있다. 그래서 그렇게 했다.
- 모델 역시 작은 걸 써야한다. 


## II. 모델 선택

### 1. Word level alignment vs. Sentence level alignment
>> **결국 Sentence Level Alignment 하기로 결정**   

- QQP의 Paperswithcode를 보니까 작은 모델들은 Word level alignment가 위에 많이 올라와 있더라
- pretrained word vector를 못쓰니까 그리고 보통 word vector pretraining도 되게 큰 corpus쓰니까 pass.
- 물론 개인적으로 sentence level modeling을 더 좋아하는 것도 한 몫 한다.


### 2. Find liteweight amongst BERT Family  
>> **ALBERT로 결정**  

- huggingface + 논문으로 쭉 살펴봄
- DistilBERT (X): 애초에 큰 BERT를 따라할 수 있는 작은 model을 만드는거라 **vocabulary를 알아야한다**. 근데 우린 안되잖아... 
- **ALBERT** (O): layer별로 weight sharing (**사이즈 작고 regularized**) + Pretraining Task 중 **Sentence Order Prediction 이 paraphrase detection으로 쉽게 재구성 가능하다**. 

### 3. 혹시 잘 안되면... BERT word vector가지고 Word Matching Model 써보지!
- https://arxiv.org/pdf/1909.00512.pdf 에 따르면 contextualized word representation 들은 word specialization 관점에서는 좋지 않아보임 (anisotrophic) --> albert word embedding이 word matching 방법론에는 좋지 않을 가능성이 있긴하지만... 
- **해보고 안좋으면 그냥 word vector도 E2E로 학습하지뭐!**

>> 여기까지 1.5일 걸렸다. 

-----------------------------------------------------------

## III. VOCAB
> 토큰은 30005번까지 있는데 실제로는 10k개도 안되어서 다시 변환해서 사용

## IV. 실험 1: ALBERT 
[github](https://github.com/sonsus/albert_paraphrase)
[wandb](https://wandb.ai/sonsus/albert_paraphrase/)
> 구현에 2일 걸렸다. 
### 3일차 실험: 성능이 너무 나쁘다, MLMLoss도 안정이 안되고.
- 아무리 데이터가 적어도... 60%를 못넘는건 말이 되나?
- 옳지! **position_id**를 안 넣었구나 (신기하게 이걸 tokentype_id (0,1 만 들어있음)로 넣고나서 **MLMLoss는 좀 안정**이 되었다고 한다) 
- ~~제출 30분 전에 안 사실이지만 position input 잘 짜놓고 거기다 token type id 넣었다고 한다 --> 여운남아서 고치고 실험 돌리는 중이다~~

### 5일차까지: 가장 중요한 Paraphrase Prediction Loss (PPLoss) 가 너무 일찍 overfit 한다. devset 성능은 ~56%
- 사이즈를 줄인다. Albert-base (12M params), Albert-large (18M) 둘 다 overfit 하는 것 같다.  
- Sentence Order Prediction loss (우리의 경우 Paraphrase Prediction Loss = PP loss) + MLM loss 1:1일 필요는 없다. 물론 그게 오리지널이긴 하지만

#### 1.Loss Balancing (alpha search) 
> Loss = alpha * PPLoss + MLMLoss

#### 2.New CLS Head
> 보통 BERT는 Encoder를 feature extractor로 freeze 하고 CLS head를 짧게 fine-tuning한다.


### 6일차: 절망 (devset0에서 계속 56%)
> 이 모든 것들을 했음에도 불구하고 성능에 큰 변화가 없다...
> 여기에 시간 더 쓰기가 어려워 보인다. 그럼에도 불구하고 복권마냥 계속 조건을 바꿔본다

- **MLMLoss만 아주 잘 내려간다 화난다**
- warmup steps
- scheduler: cosine linear
- num layers
- num attention heads
- weight decay value

-----------------------------------------------------------

## V. 실험 2: CNN-for sentence classification (Yoon et al, 2014) 
[github](https://github.com/sonsus/cnn-text-classification-pytorch)
[wandb](https://wandb.ai/sonsus/cnn-text-classification-pytorch)

### 7일차-8일차(11/16-17) 
> Papers With Code 에 나온 DIIN, RE2 코드 고쳐서 사용해보고자 했으나... 시간내에 못할 거라는 생각이 강해서 관뒀다.
[DIIN keras](https://github.com/YerevaNN/DIIN-in-Keras)
[RE2 pytorch](https://github.com/alibaba-edu/simple-effective-text-matching-pytorch)

### 마지막 6시간 (11/17) (devset0: 56%)
> 처음부터 (훨씬) 작은 모델로 했어야하는거 아니었을까? ㅠㅠ 

- 아 다 모르겠고 얘는 금방 고칠거같다 [CNN-yoon-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)
- 금방 고쳐서 21:30~23:00 까지 search

--------------------------------------------------

## VI. 제출
### 23:30~23:45 에 ALBERT 두 개, CNN 두 개 골라서 냈다
> 56% 가 아니라 74% 라서 너무 행복했다

## VII. 후기 (11/18 정리하면서)
### 그래서 baseline은 pretrained model 쓴거죠?
> 쓴거라고 말해줘요...

### 그래서 결국 k-fold cross validation은 안했다
- 성능 안나올 때 다른 devset도 돌려보면 좋았을 건데 참... condition search만 주구장창했다

### Secret test set에 대해서
- 이거 한쪽으로 라벨이 몰려있는거 아닐까? 아닌가? devset0은 반반이었다. 
- ALBERT 응답은 11/18 지금 시점에서 보니까 65~70% 정도 비율로 devset0에서 1로 응답한다.
- CNN 응답은 75~80% devset0에서 1로 응답한다
