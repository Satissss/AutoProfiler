# autokernel

LLM `{'model': 'qwen-plus', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'api_key_env': 'QWEN_API_KEY'}`, 2 rounds, benchmark `kernelbench`. Failed/missing tasks count as 1.0x in geomean_all.

## all

| method | n | correct | fast_1 [95% CI] | geomean_all [95% CI] | geomean (correct) | tokens/task | gpu s/task |
|---|---|---|---|---|---|---|---|
| autokernel | 3 | 100.0% | 0.0% [0.0, 0.0] | 0.943 [0.889, 0.985] | 0.943 | 17116 | 82 |

## level1

| method | n | correct | fast_1 [95% CI] | geomean_all [95% CI] | geomean (correct) | tokens/task | gpu s/task |
|---|---|---|---|---|---|---|---|
| autokernel | 3 | 100.0% | 0.0% [0.0, 0.0] | 0.943 [0.889, 0.985] | 0.943 | 17116 | 82 |
