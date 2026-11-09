# Events

**Module:** `reak_core.sync.events`  
**Entry points:** `process_feature_events(session_root)`, `seed_playback_annotations(session_root)`

Two steps. They do not share an output file.

## Feature and web events

Raw events are the first existing file under `rawData/`, in this order: `rawEvents.csv`, `events_0.csv`, `event.csv`. Rows are `timestamp, event name, payload`. A header is optional. A JSON payload may contain commas; the extra cells are joined back into the payload. An optional `TIMESTAMP` synchronisation event rebases the times.

These names are dropped and appear in neither output (decision D3): `RECIPE_CONFIG`, `SYSTEM_CONFIG`, `CONSENT_INFO`, `RECORDING_ONSET`.

| Event name | Feature file | Web file |
|---|---|---|
| `WEB_SCROLL`, `WEB_INITIAL_SCROLL_POSITION` | No | Yes |
| `WEB_TAB_UPDATE` | Only when the payload contains `complete` | Yes |
| `NO_KEYPRESS`, `KEYPRESS`, `ANSWER` | No | No |
| Any other name | Yes | No |

Both files are written under `features/events/` (decision D16):

- `features/events/processedFeatureEvents.csv`
- `features/events/processedWebFeatureEvents.csv`

When none of the three raw names exists, the step is skipped and writes nothing. When a raw file exists, the two outputs are written again on the next run.

## Playback annotations

`seed_playback_annotations` reads only `rawData/event.csv`. It writes `features/events/playback_annotations.json` **once**. If that file already exists, it is left untouched so edits made in playback survive the next run. Delete it to seed again.

Time zero is the `RECORDING_ONSET` row. Annotation times are seconds from that event. Without a `RECORDING_ONSET` row the step stops and writes nothing. `RECORDING_ONSET`, `RECIPE_CONFIG`, `SYSTEM_CONFIG` and `CONSENT_INFO` are not annotations. Every other row becomes a point:

```json
{
  "version": 1,
  "points": [
    { "id": "<uuid4>", "t": 12.345678, "label": "EVENT_NAME", "visible": true }
  ],
  "zones": []
}
```

`zones` starts empty. Playback can add zones later. Nucleus-Kit also copies the same JSON to `results/playback_annotations.json` when the user saves annotations in the app. The core does not write that copy.
