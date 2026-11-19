# crewai-treza

[Treza](https://www.trezalabs.com) tools for CrewAI. An agent in your crew can generate AI pictures and video clips and run your saved Treza pipelines. Treza prices every render first, and a `max_cost_usd` you set stops any render that costs more, with nothing spent.

## Install

```bash
pip install -U crewai-treza
```

Create a Treza account at https://www.trezalabs.com, add credits, and make an API key at https://www.trezalabs.com/platform/api-keys with the `pipelines:read`, `pipelines:write` and `pipelines:run` permissions. Then:

```bash
export TREZA_API_KEY="treza_live_..."
```

The package is free. Renders are paid from your prepaid Treza credits; there is no subscription.

## Tools

| Class | Name the agent sees | What it does | Cost |
| --- | --- | --- | --- |
| `TrezaGenerateImage` | `treza_generate_image` | A picture from a prompt, optionally from reference pictures. | Treza credits |
| `TrezaGenerateVideo` | `treza_generate_video` | A video clip from a prompt, optionally opening or ending on a picture. | Treza credits |
| `TrezaRunPipeline` | `treza_run_pipeline` | Runs one of your saved Treza pipelines, with new inputs if given. | Treza credits |
| `TrezaGetRun` | `treza_get_run` | Looks up a render that was still going when its tool stopped waiting. | Free |
| `TrezaBalance` | `treza_balance` | The credits left on the account. | Free |

A finished render comes back as JSON:

```json
{"url": "https://media.trezalabs.com/...mp4", "status": "success", "finished": true,
 "price_usd": 0.28, "model": "google/veo-3.1-lite", "render_seconds": 56.0,
 "run_id": "...", "pipeline_id": "pipeline_..."}
```

## Use one tool

```python
from crewai_treza import TrezaGenerateVideo

tool = TrezaGenerateVideo(max_cost_usd=1.00)
print(tool.run(prompt="A red fox trots through fresh snow", aspect_ratio="9:16", duration_sec=4))
```

## Give the tools to an agent

```python
from crewai import Agent, Crew, Task
from crewai_treza import treza_tools

director = Agent(
    role="Video director",
    goal="Turn a brief into a finished short clip",
    backstory="You write tight shot descriptions and deliver a link to the finished file.",
    tools=treza_tools(max_cost_usd=1.00),
)
task = Task(
    description="Make a 4 second vertical clip of a fox in snow.",
    expected_output="The link to the finished video and what it cost.",
    agent=director,
)
print(Crew(agents=[director], tasks=[task]).kickoff())
```

`treza_tools(include_pipelines=False)` leaves out `treza_run_pipeline`.

## Spending

- `max_cost_usd` is set in your code. The model cannot see or change it, and it cannot see the API key.
- With a cap set, Treza prices the render first. If the price is higher, the tool answers "This render costs $X, over the Max cost of $Y. Nothing was spent." and starts nothing.
- With no cap (`None`, the default), a render runs at Treza's quoted price as long as the balance covers it. **Set a cap on tools you give to an agent.**
- A render never starts when the balance does not cover its price.

## Long renders

Pictures usually take under a minute and videos one to five minutes. Each tool waits up to `wait_seconds` (300 by default). A render still going after that comes back with `"status": "running"` and its `run_id` and `pipeline_id`, not as an error, and nothing is started twice. The get-run tool returns the file once it is done.

## Errors

Refusals (over the cap, low balance, a bad key, a render that broke) go back to the agent as text, so it can react. Pass `raise_on_refusal=True` to have them raise `TrezaError` instead.

## Network and privacy

The tools call Treza's API over HTTPS at `www.trezalabs.com`. They send the prompt and settings with your API key, and store nothing. Privacy policy: https://docs.trezalabs.com/terms-and-privacy/privacy

## Development

```bash
pip install -e . pytest
pytest
```

The tests run against a scripted stand-in for Treza's API.

## Support

Issues: https://github.com/treza-labs/crewai-treza/issues · Email: hello@trezalabs.com

MIT licensed.
