# kernelagent

LLM `{'model': 'qwen-plus', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'api_key_env': 'QWEN_API_KEY'}`, 2 rounds, benchmark `kernelbench`. Failed/missing tasks count as 1.0x in geomean_all.

## all

| method | n | correct | fast_1 [95% CI] | geomean_all [95% CI] | geomean (correct) | tokens/task | gpu s/task |
|---|---|---|---|---|---|---|---|
| kernelagent | 3 | 100.0% | 33.3% [0.0, 100.0] | 1.181 [0.889, 1.874] | 1.181 | 31047 | 66 |

## level1

| method | n | correct | fast_1 [95% CI] | geomean_all [95% CI] | geomean (correct) | tokens/task | gpu s/task |
|---|---|---|---|---|---|---|---|
| kernelagent | 3 | 100.0% | 33.3% [0.0, 100.0] | 1.181 [0.889, 1.874] | 1.181 | 31047 | 66 |
