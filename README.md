# irunner

Бюджетно-осознанный харнесс на openJiuwen для задач с общим лимитом токенов.

## Запуск

```bash
docker build -t hack:latest .

docker run --rm --network hack-net \
  -e LLM_BASE_URL -e LLM_API_KEY -e BUDGET \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --workdir /app --entrypoint python \
  hack:latest /app/run_agent.py --tasks-dir /data/tasks
```

- `BUDGET` — бюджет в выходных токенах; если не задан, `100000`.
  Поддерживаются `100000`, `100_000`, `100K`, `1M`.
- При отсутствии `OUT` файлы решений пишутся в `/data/tasks/.irunner`.
- `answer.txt` создаётся рядом с `task.json`; для невыполнимых задач — не создаётся.

## Как работает

Сначала бесплатная фаза: детерминированные math/data/search, mutation repair
для coding и детекторы невыполнимых задач. Остаток распределяет планировщик
(теневая цена бюджета + 0/1-рюкзак), затем LLM-fallback через
`openjiuwen.core.foundation.llm.Model`. Бюджет считается как
`output_tokens + input_tokens / 10`; при исчерпании прокси харнесс сохраняет
уже полученные ответы.

## Проверки

```bash
python3 -m unittest discover -s tests
```

Примеры траекторий: `traces/example_trace.jsonl`.
