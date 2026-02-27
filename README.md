# LLM Judge Benchmark

> **LLM-as-a-Judge** 방식으로 LLM의 응답 품질을 자동 평가하는 벤치마크 파이프라인

## 프로젝트 개요

본 프로젝트는 **대상 LLM(Qwen 2.5-72B)**이 생성한 답변을 **심판 LLM(GPT-5.1)**이 자동으로 채점하는 **LLM-as-a-Judge** 평가 시스템입니다.

주거 환경(실내 온도, 습도, CO₂, VOC, PM2.5 등) 및 건강(심박수, 호흡수, 활동량) 도메인의 센서 데이터를 분석하여, Qwen이 생성한 현황 요약·권고문의 품질을 GPT가 정량적으로 평가합니다.

### 평가 지표

| 지표 | 설명 |
|------|------|
| **Faithfulness (사실충실도)** | LLM 답변이 근거 자료(`retrieved_result`)와 사실적으로 얼마나 일치하는지 (1~5점) |
| **Relevance (관련성)** | LLM 답변이 근거 자료의 핵심 초점에 얼마나 집중하는지 (1~5점) |

---

## 아키텍처

![alt text](<시스템 아키텍처.png>)

### 파이프라인 실행 흐름 (통합 버전)

![alt text](<파이프라인 실행 흐름.png>)

---

## 디렉토리 구조

```
llm-judge-benchmark/
│
├── benchmark_pipeline_sync.py                      # 동기식 파이프라인
├── benchmark_pipeline_async.py                     # 비동기식 파이프라인
├── benchmark_pipeline_async_parallel.py            # 비동기 병렬 파이프라인 (GPT 평가만)
├── benchmark_pipeline_async_parallel_llm_answer.py # 비동기 병렬 통합 (Qwen 답변 생성 + GPT 평가)
│
├── config.yaml              # 전체 설정 (경로, 동시 호출 수, 반복 횟수 등)
├── settings.env             # API 키 및 서버 URL
├── requirements.txt         # Python 의존성
│
├── common/
│   ├── config/loader.py     # config.yaml 로딩
│   └── utils/io.py          # 텍스트 파일 로딩
│
├── gpt/
│   ├── llm_client.py        # AsyncOpenAI 클라이언트
│   └── async_scorer.py      # GPT 비동기 채점 (score_row, 병렬 처리)
│
├── qwen/
│   └── model.py             # Qwen 모델 호출 (ModelManager), 임베딩, 리랭커
│
├── calculate_function/
│   └── calculate_functions.py  # 점수 집계 (Faithfulness, Relevance)
│
├── prompt/
│   ├── system_prompt_faithfulness.txt       # GPT 심판: 사실충실도 평가 프롬프트
│   ├── system_prompt_relevance.txt          # GPT 심판: 관련성 평가 프롬프트
│   ├── user_template.txt                    # GPT 심판: 입력 템플릿
│   ├── system_prompt_qwen_environment.txt   # Qwen: 주거환경 분석 프롬프트
│   ├── system_prompt_qwen_health.txt        # Qwen: 건강 분석 프롬프트
│   ├── user_template_qwen_environment.txt   # Qwen: 주거환경 입력 템플릿
│   └── user_template_qwen_health.txt        # Qwen: 건강 입력 템플릿
│
├── data/
│   └── test_dataset/
│       ├── environment_50.csv   # 주거환경 테스트 데이터 (50건)
│       ├── health_50.csv        # 건강 테스트 데이터 (50건)
│       └── output/              # 평가 결과 저장 디렉토리
│           ├── llm_answer/      #   Qwen 응답 캐시
│           └── scores/          #   점수 요약 파일
│
└── log/                         # 실행 로그
```

---

## 설치 및 환경 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`settings.env` 파일에 다음 값을 설정합니다:

```env
OPENAI_KEY='your-openai-api-key'
QWEN_API='your-qwen-model-endpoint-url'
```

### 3. 설정 파일 (`config.yaml`)

```yaml
llm:
    max_limit_qwen: 15        # Qwen 동시 호출 수

paths:
    prompt_dir: 'prompt'
    dataset_dir: 'data/test_dataset'
    log_dir: 'log'
    output_dir_name: 'output'
    score_dir_name: 'scores'
    llm_answer_dir_name: 'llm_answer'

datasets:
    env_csv: 'environment_50.csv'
    health_csv: 'health_50.csv'

eval:
    max_limit_gpt: 25         # GPT 동시 호출 수
    repetition_num: 3         # 평가 반복 횟수

model:
    name: 'gpt-5.1'           # 심판 모델
```

---

## 실행 방법

### 권장: 비동기 병렬 통합 파이프라인

Qwen 답변 생성부터 GPT 평가까지 한 번에 실행합니다.

```bash
python benchmark_pipeline_async_parallel_llm_answer.py
```

### 기타 파이프라인

```bash
# GPT 평가만 실행 (llm_answer 컬럼이 이미 CSV에 존재하는 경우)
python benchmark_pipeline_async_parallel.py

# 비동기 단일 처리
python benchmark_pipeline_async.py

# 동기식 처리
python benchmark_pipeline_sync.py
```

### 실행 결과

실행이 완료되면 다음 파일들이 생성됩니다:

| 출력 파일 | 위치 | 설명 |
|-----------|------|------|
| `*_scored_run{i}.csv` | `data/test_dataset/output/` | 각 행에 score, scoring_reason 컬럼이 추가된 CSV |
| `*_llm_answer.csv` | `data/test_dataset/output/llm_answer/` | Qwen이 생성한 답변이 포함된 CSV (재사용 가능) |
| `faithfulness_relevance_scores{i}.txt` | `data/test_dataset/output/scores/` | 점수 집계 요약 |
| `*.log` | `log/` | 실행 로그 |

---

## 설계 시 고민했던 점

### 1. LLM-as-a-Judge 평가 방식 선택 이유

주거 환경 데이터 50건, 건강 데이터 50건 — 총 **100건의 LLM 응답**에 대해 품질 평가를 수행해야 했습니다. 이를 전문가를 고용하여 수동으로 평가하는 방식은 다음과 같은 이유로 적합하지 않았습니다:

- **비용 비효율성**: 도메인 전문가가 100건의 응답을 하나하나 읽고 채점하는 데 상당한 인건비와 시간이 소요됩니다.
- **반복 평가의 필요성**: 평가 점수의 신뢰도를 확보하기 위해 동일 데이터에 대해 **여러 차례 반복 평가**를 수행하고 평균을 산출해야 했습니다. 사람이 이를 반복하는 것은 현실적으로 불가능에 가깝습니다.
- **일관된 기준 적용**: 사람마다 채점 기준이 달라질 수 있지만, LLM 심판은 동일한 프롬프트를 기반으로 **일관된 기준**을 적용할 수 있습니다.

이러한 배경에서 **GPT-5.1을 심판 LLM으로 활용**하는 LLM-as-a-Judge 방식을 채택하여, 짧은 시간 내에 자동화된 반복 평가가 가능하도록 했습니다.

### 2. 파이프라인의 단계적 발전

초기 평가 계획에서는 Qwen 모델의 응답(`llm_answer`)을 **사전에 생성·저장**해 두고, 평가 시에는 저장된 응답에 대해 GPT 채점만 수행하는 구조였습니다.

그러나 담당 평가관과의 소통을 통해, **평가 자리에서 답변 생성부터 평가까지 전 과정이 실시간으로 진행**되어야 한다는 요건을 확인했습니다. 이에 따라 파이프라인을 다음과 같이 발전시켰습니다:

```
sync → async → async parallel → async parallel + llm_answer 통합
```

| 단계 | 개선 내용 | 효과 |
|------|----------|------|
| **sync** | 동기식 순차 처리 | 기본 동작 검증 |
| **async** | GPT API 비동기 호출 | 단일 요청 대기 시간 제거 |
| **async parallel** | ENV + HEALTH 두 데이터셋 동시 평가 | 100건을 하나의 진행바로 동시 처리 |
| **async parallel + llm_answer** | Qwen 답변 생성과 GPT 평가를 하나의 파이프라인으로 통합 | 평가 현장에서 전 과정 자동 실행 |

특히, **하루라는 제한된 시간** 내에 100건의 답변 생성 + 평가 + 일정 횟수 반복까지 완료해야 했기 때문에, OpenAI의 `AsyncOpenAI` 패키지를 활용한 **비동기 병렬 처리**가 필수적이었습니다. 이를 통해 짧은 시간 내에 응답 생성, 평가, 점수 집계까지 일괄 수행할 수 있도록 했습니다.

### 3. 동시성 제어 (Concurrency Control)

비동기 병렬 처리를 도입하면서, API Rate Limit 초과로 인한 요청 실패를 방지하는 것이 중요한 과제였습니다. `asyncio.Semaphore`를 활용하여 동시 호출 수를 제한함으로써, 서버 안정성과 처리 속도 사이의 균형을 맞추었습니다:

| 대상 | 설정값 | 제한 이유 |
|------|--------|----------|
| **Qwen** | `max_limit_qwen = 15` | 로컬 GPU 서버의 동시 추론 부하 고려 |
| **GPT** | `max_limit_gpt = 25` | OpenAI API Rate Limit 초과 방지 |

또한, GPT 채점 실패 시 **최대 3회 재시도**(retry) 로직을 적용하여 일시적인 네트워크 오류나 타임아웃에도 안정적으로 전체 평가를 완료할 수 있도록 했습니다.

### 4. LLM 응답 캐싱

Qwen 답변 생성은 GPU 자원과 시간이 소요되는 고비용 작업입니다. 따라서 한 번 생성된 `llm_answer`는 CSV 파일로 캐시(`llm_answer/` 디렉토리)하여, 동일한 답변을 중복 생성하지 않도록 했습니다.

이후 반복 평가(`repetition_num=3`)에서는 **캐시된 동일 LLM 답변**에 대해 GPT 채점만 반복 수행합니다. 이를 통해:

- **평가 시간 단축**: Qwen 호출을 생략하여 반복 평가 소요 시간을 대폭 절감합니다.
- **채점 일관성 측정**: 동일한 입력에 대한 GPT의 채점 결과가 회차별로 얼마나 안정적인지 확인할 수 있습니다.

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| **언어** | Python 3.10+ |
| **대상 LLM** | Qwen 2.5-72B-Instruct (로컬 vLLM 서버) |
| **심판 LLM** | GPT-5.1 (OpenAI API) |
| **비동기 처리** | asyncio, AsyncOpenAI |
| **데이터 처리** | pandas |
| **설정 관리** | PyYAML, python-dotenv |
| **진행 표시** | tqdm |

---

## 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
