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

Time zero is the `RECORDING_ONSET` row. Annotation times are seconds from that event. Without a `RECORDING_ONSET` row the step stops and writes nothing. `RECORDING_ONSET`, `RECIPE_CONFIG`, `SYSTEM_CONFIG` and `CONSENT_INFO` are not annotations. Paired zone tags become zones ([Recorded zones](#recorded-zones)); every other row becomes a point:

```json
{
  "version": 1,
  "points": [
    { "id": "<uuid4>", "t": 12.345678, "label": "EVENT_NAME", "visible": true }
  ],
  "zones": []
}
```

`zones` holds the recorded zones (below); playback can add more. Nucleus-Kit also copies the same JSON to `results/playback_annotations.json` when the user saves annotations in the app. The core does not write that copy.

## Recorded zones

Prime's live zones send two tags to every recording phone, which writes them to `event.csv`: `ZONE_START` and
`ZONE_STOP`, both with `Values` = `<zoneId>|<name>`. A start and a stop with the same id make one zone, labelled with
the start's name. Zones may overlap. A value without `|` is both the id and the name.

- **Playback annotations:** each pair becomes a zone `{ "id", "t0", "t1", "label", "visible" }` (seconds from
  `RECORDING_ONSET`). A start without a stop, or a stop without a start, stays a point.
- **`recording_zones(session_root)`:** reads the first raw event file found (`rawEvents.csv`, `events_0.csv`,
  `event.csv`; the session API's `t, 'NAME', value` rows are accepted) and returns
  `[{"zone_id", "label", "start_s", "end_s"}]` in seconds from the **first EXG sample**, the time base of the
  results files.
  - A zone still open when the recording stops ends with the recording; a stop whose start came before the recording
    begins at 0. Zones are clipped to the recording.
  - `[]` when the session has no zone tag (the EXG is not read).
  - `None` when the zones cannot be placed: no EXG file, or an event or EXG clock that is not Unix time.

Prime publishes these zones to the website as `recording` zone annotations (primeV3 `docs/pipeline_events.md`).
