# 그래프 기반 적응형 스키마 링킹을 통한 Text-to-SQL 성능 개선

**Graph-based Adaptive Schema Linking for Text-to-SQL**

> 한국지능정보시스템학회 2026 춘계 학술대회 — Extended Abstract 초안
> (본문 최대 3 pages, 표지 포함 4 pages 제약)

---

## 메타 정보 (템플릿 상단 블록)

- **저자/소속/이메일** *(확인 필요)*
  - 주저자: {이름} (연세대학교 {소속}, {이메일})
  - 공동저자: {이름} ({소속}, {이메일})
  - 교신저자: {이름} ({소속}, {이메일})
- **Keywords (국문, 가나다 순, 5개)**: 그래프 신경망, 스키마 링킹, 적응형 부분그래프 추출, 질의응답, Text-to-SQL
- **Keywords (영문, alphabetical)**: Adaptive Subgraph Extraction, Graph Neural Network, Retrieval-Augmented Generation, Schema Linking, Text-to-SQL
- **사사표기**: *{해당 연구과제 번호 기입, 없으면 생략}*

---

## Abstract

Text-to-SQL에서 대규모 데이터베이스의 **Schema Linking**은 SQL 생성 품질을 좌우하는 핵심 전처리 단계이다. 그러나 기존 연구는 (i) 스키마의 관계적 구조를 반영하지 못하는 벡터 검색, (ii) 질의별 관련도 분포의 편차를 포착하지 못하는 고정 임계치 기반 Graph-RAG, (iii) 구조 정보를 직접 활용하지 못하는 LLM 에이전트 필터의 세 가지 한계 축에 각각 머물러 있으며, 이들을 **구조-적응성-의미 검증**의 세 수준에서 일관되게 결합한 접근은 부족한 실정이다. 본 논문은 이종 스키마 그래프 위에서 동작하는 **6-Stage 적응형 Schema Linking 파이프라인**을 제안한다. 제안 기법은 (1) 사전학습 문장 인코더의 의미 신호와 GATv2 기반 구조 신호를 앙상블하는 **Ensemble Seed Selector**, (2) 질의별 점수 분포에 따라 Prize 임계치를 동적으로 결정하여 Prize-Collecting Steiner Tree의 과선택·과삭제 문제를 완화하는 **Adaptive PCST Extractor**, (3) 선택된 테이블 쌍의 외래키 컬럼을 자동으로 보강하여 JOIN 경로 누락을 방지하는 **Auto JOIN Key Injection**, (4) 데이터베이스 값 예시를 프롬프트에 주입하여 LLM이 컬럼의 실제 의미까지 검증하도록 유도하는 **XiYan-style Value-aware Filter**로 구성된다. 또한 Seed Selector의 GAT 모듈은 포인트별 이진 교차 엔트로피 손실과 하드 네거티브 기반 InfoNCE 손실을 결합하여, 노드별 보정과 후보 집합 수준의 변별력을 동시에 학습하도록 설계된다. BIRD-Dev 벤치마크에 대한 예비 실험에서 제안 기법은 대표 baseline들과 비교했을 때 높은 정밀도와 간결한 선택 집합 측면의 경향을 확인하였으며, 이는 그래프 기반 구조 활용과 질의 적응성이 Schema Linking 품질 개선에 기여할 수 있음을 시사한다.

**Keywords** — 그래프 신경망, 스키마 링킹, 적응형 부분그래프 추출, 질의응답, Text-to-SQL

---

## I. 서론

### 1.1 연구 배경

자연어로 데이터베이스를 질의하는 **Text-to-SQL**은 비전문가의 데이터 접근성을 크게 개선하는 기술로, Spider (Yu et al., 2018), BIRD (Li et al., 2023), WikiSQL (Zhong et al., 2017) 등 대규모 벤치마크의 등장과 함께 빠르게 발전해 왔다. 특히 최근에는 대형 언어 모델(Large Language Model, LLM)의 발전 (Brown et al., 2020; OpenAI, 2023)에 힘입어 문장 생성 기반 Text-to-SQL 접근이 주류를 이루고 있으며 (Rajkumar et al., 2022; Pourreza & Rafiei, 2023), 실행 정확도(Execution Accuracy) 기준으로 인간 전문가 수준에 근접하는 성능이 보고되고 있다 (Gao et al., 2024).

그러나 Text-to-SQL 파이프라인의 실질적 성능은 단순히 언어 모델의 생성 능력에만 의존하지 않는다. 실무 데이터베이스는 일반적으로 수십에서 수백 개의 테이블과 수천 개의 컬럼을 포함하며 (Li et al., 2023), 테이블 간에는 외래키 관계와 같은 복잡한 참조 구조가 존재한다. 이러한 대규모 스키마를 LLM 컨텍스트에 모두 제공하는 것은 토큰 비용과 지연시간 측면에서 비효율적일 뿐만 아니라, 무관한 스키마 요소가 오히려 환각(hallucination) 현상과 잘못된 JOIN 경로 선택을 유발하는 원인으로 보고된다 (Ji et al., 2023; Dong et al., 2023).

이에 따라 SQL 생성 이전에 질의와 관련된 테이블·컬럼만을 사전에 식별하는 **Schema Linking**이 Text-to-SQL의 핵심 전처리 단계로 주목받고 있다 (Wang et al., 2020; Lei et al., 2020). Schema Linking은 본질적으로 **의미적 관련성**(질의와 스키마 요소의 의미 대응)과 **구조적 연결성**(선택된 요소들이 SQL로 집행 가능한 서브스키마를 구성할 수 있는지)이라는 두 가지 제약을 동시에 만족해야 하는 문제이며, 이 두 제약 사이의 균형을 어떻게 설계하는가가 연구의 핵심 쟁점이다 (Cao et al., 2021).

### 1.2 Schema Linking 연구의 흐름

Schema Linking 연구는 Text-to-SQL 전반의 발전 양상과 맞물려 **의미 기반 검색**, **구조 기반 검색**, 그리고 최근의 **LLM 기반 추론**이라는 세 흐름을 차례로 거쳐 왔다.

초기 연구는 질의와 스키마 요소를 동일한 임베딩 공간에 투영하여 의미적 유사도로 후보를 선별하는 방향을 중심으로 발전하였다 (Reimers & Gurevych, 2019; Karpukhin et al., 2020). 이후 데이터베이스 스키마가 본질적으로 **관계형 그래프 구조**를 가진다는 점에 주목하여, 그래프 신경망 (Kipf & Welling, 2017; Veličković et al., 2018; Brody et al., 2022)이나 Prize-Collecting Steiner Tree (Goemans & Williamson, 1995; Hegde et al., 2015)와 같은 그래프 기반 알고리즘을 활용해 관련 요소를 연결된 부분그래프로 추출하려는 흐름이 부상하였다 (Wang et al., 2020; Cao et al., 2021; He et al., 2024). 가장 최근에는 대형 언어 모델의 추론 능력에 의존하여 후보 요소를 반복적으로 검증·정제하는 **에이전트 기반** 접근이 활발히 탐구되고 있으며 (Pourreza & Rafiei, 2023; Wang et al., 2024; Talaei et al., 2024), 외부 도구 및 스키마 메타정보를 결합한 다단계 프레임워크가 보고되고 있다 (Lee et al., 2025; Gao et al., 2024).

이러한 세 흐름은 각기 다른 축에 강점을 두고 있다. 의미 기반 접근은 **의미적 관련성**을, 그래프 기반 접근은 **구조적 연결성**을, LLM 기반 접근은 **문맥적 의미 검증**을 각각 최적화한다. 그러나 이들 접근은 주로 단일 축에 집중되는 경향이 있어, Schema Linking이 본질적으로 요구하는 **의미 · 구조 · 검증**의 세 축을 하나의 일관된 파이프라인 안에서 유기적으로 결합하는 시도는 상대적으로 부족한 것으로 관찰된다. 또한 그래프 기반 접근에서는 **질의 적응성**(query-adaptivity) — 즉, 질의마다 달라지는 관련도 점수 분포를 능동적으로 반영하는 메커니즘 — 이 충분히 고려되지 않는 경향이 있으며 (He et al., 2024), LLM 기반 접근에서는 스키마 메타정보만으로는 드러나지 않는 **실제 데이터 분포**를 검증 과정에 통합하는 방법이 아직 활발히 논의되고 있는 단계이다 (Gao et al., 2024; Talaei et al., 2024). 개별 연구에 대한 구체적인 분석은 II장에서 다룬다.

### 1.3 연구 방향 및 기여

본 논문은 위와 같은 연구 동향에서 관찰되는 **축 간 결합의 부재**와 **질의 적응성 부족**이라는 두 공백에 주목한다. 이를 위해 이종 스키마 그래프 위에서 동작하는 **6-Stage 적응형 Schema Linking 파이프라인**을 제안하며, Schema Linking 문제를 **의미 검색 → 구조 기반 확장 → 질의 적응적 가지치기 → 값 기반 의미 검증**이라는 네 단계로 분해하여 각 단계에 대응하는 구조적 기여를 배치한다.

1. **Ensemble Seed Selector**: 사전학습 문장 인코더 (Reimers & Gurevych, 2019)의 의미 신호와 그래프 신경망 (Brody et al., 2022) 기반의 구조 신호를 단일 점수 공간에서 결합하여, 의미·구조 두 축의 상호 보완을 가능하게 한다. 학습 단계에서는 포인트별 보정을 담당하는 이진 교차 엔트로피 손실과, 후보 집합 수준의 변별력을 담당하는 하드 네거티브 기반 InfoNCE 손실 (van den Oord et al., 2018; Robinson et al., 2021)을 결합하여, pointwise 보정과 listwise 랭킹 학습을 동시에 수행한다.

2. **Adaptive PCST Extractor**: Prize-Collecting Steiner Tree 기반 부분그래프 추출 (Goemans & Williamson, 1995; He et al., 2024) 과정에 **질의 적응성**을 부여한다. 질의별 점수 분포에서 계산된 백분위수를 동적 임계치로 사용하여 Prize를 산출하고, 질의 난이도에 따라 부분그래프 크기가 자연스럽게 조절되도록 한다.

3. **Auto JOIN Key Injection**: 선택된 테이블 쌍이 공유하는 외래키 컬럼을 후처리 단계에서 자동으로 보강하여, 의미적으로 드러나지 않는 연결 컬럼의 누락으로 인한 JOIN 경로 실패를 방지한다.

4. **Value-aware LLM Filter**: 최종 단계에 실제 데이터베이스 값 예시를 프롬프트에 주입한 LLM 필터 (Gao et al., 2024)를 배치하여, 스키마 메타정보만으로는 판별이 어려운 컬럼의 **데이터 수준 의미 검증**을 가능하게 한다.

이러한 네 가지 구조적 기여는 각 단계에서 개별적으로 동작하는 것을 넘어, 상위 단계에서 포착한 후보 집합을 하위 단계가 보완·검증하는 **계층적 정제 구조**를 형성한다. 이는 Schema Linking을 단일 모델의 end-to-end 예측 문제가 아닌, **의미·구조·검증의 세 축이 상호작용하는 파이프라인 수준의 설계 문제**로 재정의하는 관점을 제시한다.

### 1.4 논문 구성

본 논문의 나머지는 다음과 같이 구성된다. II장에서는 Text-to-SQL 및 Schema Linking 관련 선행 연구를 상세히 검토한다. III장에서는 제안하는 6-Stage 파이프라인의 각 모듈을 기술하며, IV장에서는 BIRD-Dev 벤치마크에 대한 예비 실험 결과와 ablation 분석을 제시한다. 마지막으로 V장에서 연구의 의의와 향후 연구 방향을 논의한다.

---

## 인용 참고문헌 (서론 초안 기준, 잠정)

본 초안의 I. 서론에서 인용된 문헌을 정리한다. 정식 포맷은 Extended Abstract 최종본의 참고문헌 장에서 *지능정보연구* 스타일로 재정비한다. 서지 정보 일부는 확인/보정 필요(†표시).

**벤치마크 및 Text-to-SQL 전반**
- Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., ... & Radev, D. (2018). Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and Text-to-SQL task. *EMNLP.*
- Zhong, V., Xiong, C., & Socher, R. (2017). Seq2SQL: Generating structured queries from natural language using reinforcement learning. *arXiv:1709.00103.*
- Li, J., Hui, B., Qu, G., Yang, J., Li, B., Li, B., ... & Li, Y. (2023). Can LLM already serve as a database interface? A BIG Bench for large-scale database grounded Text-to-SQLs. *NeurIPS.*

**LLM 및 Text-to-SQL 생성 모델**
- Brown, T. B., et al. (2020). Language models are few-shot learners. *NeurIPS.*
- OpenAI. (2023). GPT-4 technical report. *arXiv:2303.08774.* †
- Rajkumar, N., Li, R., & Bahdanau, D. (2022). Evaluating the Text-to-SQL capabilities of large language models. *arXiv:2204.00498.*
- Pourreza, M., & Rafiei, D. (2023). DIN-SQL: Decomposed in-context learning of Text-to-SQL with self-correction. *NeurIPS.*
- Gao, D., Wang, H., Li, Y., Sun, X., Qian, Y., Ding, B., & Zhou, J. (2024). Text-to-SQL empowered by large language models: A benchmark evaluation. *VLDB.* †

**LLM 환각·오류 분석**
- Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys, 55(12), 1-38.*
- Dong, X., et al. (2023). C3: Zero-shot Text-to-SQL with ChatGPT. *arXiv:2307.07306.* †

**Schema Linking 관련 선행 연구**
- Wang, B., Shin, R., Liu, X., Polozov, O., & Richardson, M. (2020). RAT-SQL: Relation-aware schema encoding and linking for Text-to-SQL parsers. *ACL.*
- Lei, W., Wang, W., Ma, Z., Gan, T., Lu, W., Kan, M.-Y., & Chua, T.-S. (2020). Re-examining the role of schema linking in Text-to-SQL. *EMNLP.*
- Cao, R., Chen, L., Chen, Z., Zhao, Y., Zhu, S., & Yu, K. (2021). LGESQL: Line graph enhanced Text-to-SQL model with mixed local and non-local relations. *ACL.*
- He, X., Tian, Y., Sun, Y., Chawla, N. V., Laurent, T., LeCun, Y., Bresson, X., & Hooi, B. (2024). G-Retriever: Retrieval-augmented generation for textual graph understanding and question answering. *NeurIPS.*
- Wang, B., Ren, C., Yang, J., Liang, X., Bai, J., Chai, L., ... & Li, Z. (2024). MAC-SQL: A multi-agent collaborative framework for Text-to-SQL. *arXiv:2312.11242.*
- Talaei, S., Pourreza, M., Chang, Y.-C., Mirhoseini, A., & Saberi, A. (2024). CHESS: Contextual harnessing for efficient SQL synthesis. *arXiv:2405.16755.* †
- Lee, et al. (2025). LinkAlign: Scalable schema linking for real-world large-scale multi-database Text-to-SQL. *arXiv.* † (서지 확인 필요)

**임베딩 및 검색**
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP.*
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-T. (2020). Dense passage retrieval for open-domain question answering. *EMNLP.*

**그래프 신경망**
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR.*
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *ICLR.*
- Brody, S., Alon, U., & Yahav, E. (2022). How attentive are graph attention networks? *ICLR.*

**PCST 알고리즘**
- Goemans, M. X., & Williamson, D. P. (1995). A general approximation technique for constrained forest problems. *SIAM Journal on Computing, 24(2), 296-317.*
- Hegde, C., Indyk, P., & Schmidt, L. (2015). A nearly-linear time framework for graph-structured sparsity. *ICML.*

**대조 학습 및 하드 네거티브**
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv:1807.03748.*
- Robinson, J., Chuang, C.-Y., Sra, S., & Jegelka, S. (2021). Contrastive learning with hard negative samples. *ICLR.*

*(II~V장 확장 시 필요한 추가 인용은 이후 iteration에서 보완)*

---

## 사용자 확인 필요 항목

- [ ] 저자/소속/이메일 (주저자·공동저자·교신저자 표기)
- [ ] 사사표기 대상 연구과제 유무
- [ ] 영문 제목 확정
- [ ] 초록에 ablation(EXP-C F1=0.7863)을 강조할지, Full 모델 Precision만 전면에 내세울지 톤 결정
- [ ] 키워드 5개 확정
