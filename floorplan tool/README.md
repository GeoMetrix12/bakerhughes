
# Smart Building Ops Center
Centralized HVAC, lighting, and insights driven by multi-floor occupancy, ambient sensors, and weather. Includes a professional Streamlit dashboard and optional local LLM (Ollama) for JSON operational briefs. This README combines a detailed design document and user guide in one place.

---

## 1. Problem, Challenges, and Goals

### Market Problem
Commercial buildings waste 10–30% of energy due to static schedules, poor space consolidation, and limited real-time context around occupancy and weather.

### Real-World Challenges
- Dynamic occupancy with static controls
- Fragmented systems (BMS, IoT, meters, cameras)
- Weather and tariff volatility
- Data gaps and sensor noise
- Privacy constraints and varying HVAC topologies

### Project Goals
- Convert occupancy, ambient sensors, and weather forecasts into actionable HVAC/lighting decisions
- Consolidate small groups into common rooms to reduce active area where feasible
- Provide an operator-friendly, single-pane dashboard with KPIs, trends, and forecasts
- Keep costs and privacy risk low using local LLM and free weather APIs
- Make the pipeline production-shaped but runnable entirely with synthetic data

---

## 2. Features

- Multi-floor building model with geometry and metadata (capacity, area, zone, common-room flag)
- Synthetic generators for occupancy, sensors (CO₂, lux, noise, motion, door, RH), weather, and energy by floor/building
- Merge policy (floor and zone aware) to consolidate non-common rooms into common rooms
- Control policy with daylight dimming, CO₂ ventilation boost, door-open eco mode, and vacancy shutoff
- Command emission to JSONL log and optional MQTT topics
- Professional Streamlit dashboard: floor plans, KPIs, trends, weather forecast, and AI JSON brief
- Free local LLM via Ollama for strictly JSON operational briefs; no external keys required

---

## 3. Tech Stack

- Python 3.9–3.12
- Streamlit (UI), Plotly (visualizations)
- Pandas, NumPy
- Requests (Open-Meteo API)
- Ollama (local LLM, optional)
- paho-mqtt (optional IoT egress)

No GPU required. CCTV/YOLO are not used in this synthetic-only build.

---

## 4. Repository Structure

```

data/
floors.json               # Multi-floor plan with room geometry and attributes
outputs/
synth/
occupancy.csv           # Synthetic occupancy timeline
sensors.csv             # Synthetic ambient sensors
meter\_floor.csv         # Energy by floor (kW)
meter\_building.csv      # Energy for entire building (kW)
weather\_synth.csv       # Synthetic outdoor temperature
src/
**init**.py
floors.py                 # Floorplan loader and helpers
synth.py                  # Synthetic data generator (occupancy, sensors, meters)
merge\_policy.py           # Floor+zone aware merge suggestions
policy.py                 # Control policy deriving device commands
open\_meteo.py             # Geocoding + hourly forecast + summary bullets
llm\_local.py              # Local LLM (Ollama) JSON brief
iot.py                    # Command egress (JSONL and optional MQTT)
streamlit\_app.py            # Professional dashboard
requirements.txt
DESIGN.md                   # Optional separate design doc (this README already contains details)
README.md

````

---

## 5. Architecture and Data Flow

### Logical Components
- Data model: `data/floors.json` (floors, rooms, geometry, attributes)
- Synthetic data: `src/synth.py` produces `outputs/synth/*.csv`
- Decision layer:
  - `src/merge_policy.py` for consolidation (per floor, per zone)
  - `src/policy.py` for HVAC/lighting actions given counts, sensors, and assignments
- Weather adapter: `src/open_meteo.py` (geocode + forecast + bullets)
- LLM adapter: `src/llm_local.py` (Ollama JSON brief)
- IoT egress: `src/iot.py` (writes JSONL, optional MQTT)
- UI: `streamlit_app.py` (custom CSS, Plotly, tabs)

### High-Level Flow

```mermaid
flowchart LR
    A[Floors (floors.json)] --> B[Synth Generator (synth.py)]
    B -->|occupancy.csv, sensors.csv, meter data| C[Streamlit UI]
    D[Open-Meteo API] --> C
    C -->|counts + sensors snapshot| E[Merge Policy]
    E -->|assignments + suggestions| F[Control Policy]
    F -->|commands JSONL + MQTT| G[IoT Devices / Logs]
    C -->|Ollama prompt| H[Local LLM (Ollama)]
    H -->|JSON brief| C

````

### Control Loop at a Timepoint

```mermaid
sequenceDiagram
    participant UI as Dashboard
    participant OCC as occupancy.csv
    participant SEN as sensors.csv
    participant MP as merge_policy
    participant CP as policy
    participant IOT as iot.py

    UI->>OCC: Load counts (timestamp = t)
    UI->>SEN: Load sensors (timestamp = t)
    UI->>MP: suggest_merges(counts_by_floor)
    MP-->>UI: assignments + suggestions + saved_area
    UI->>CP: derive_commands(floors, counts, assignments, sensors)
    CP-->>UI: commands[]
    UI->>IOT: (optional future: live dispatch)
```

---

## 6. Data Schemas

`occupancy.csv`
timestamp (ISO), floor\_id, room\_id, count\:int

`sensors.csv`
timestamp, floor\_id, room\_id, co2\:float, lux\:float, noise\:float, motion\:int{0,1}, door\:int{0,1}, rh\:float

`meter_floor.csv`
timestamp, floor\_id, meter\_kw\:float

`meter_building.csv`
timestamp, meter\_kw\:float

`floors.json` (excerpt)

```json
{
  "floors": [
    {
      "floor_id": "F1",
      "name": "Ground Floor",
      "rooms": [
        {"room_id":"Lobby","area_m2":160,"capacity":80,"zone":"Z1","is_common":true,"x":0,"y":0,"w":4,"h":2}
      ]
    }
  ]
}
```

---

## 7. Algorithms, Assumptions, and Key Decisions

### Synthetic Generator

* Occupancy: Binomial(capacity, p(hour, is\_common)) with inertia (EMA, alpha=0.5)
* Sensors:

  * CO₂ ppm ≈ 450 + 35 × people ± N(0, 40), clipped 400–2000
  * Lux higher in daytime; dimming threshold at 600 lux
  * Noise ≈ 35 + 3 × sqrt(people) ± N(0,2)
  * Motion, door, RH modeled stochastically within realistic bounds
* Energy:

  * Per-floor kW = base\_kw/n\_floors + 0.001 × per\_person\_w × people + hvac\_coeff × max(T − setpoint, 0) × active\_area\_share
  * Building kW = sum of floor kW

### Merge Policy (`merge_policy.py`)

* Floor- and zone-scoped consolidation of non-common rooms into common rooms
* Greedy fill of smallest common capacities first
* Output:

  * assignments\[floor\_id]\[room\_id] = { assigned\:int, active\:bool }
  * suggestions\[] = { floor\_id, from, moves\[], zone, note }
  * saved\_area\_m2: float

### Control Policy (`policy.py`)

* count == 0  => lights\:off, fan\:off, hvac\:off
* occupied    => lights\:on (dim if lux ≥ 600), fan\:on
* inactive by merge => hvac\:eco
* CO₂ ≥ 1200 ppm => hvac\:comfort with vent\_boost flag
* door open => hvac\:eco + advisory

### Weather

* Open-Meteo geocoding + hourly forecast
* Summary bullets: hot hours, rainy hours, windy hours

### LLM (Ollama)

* Strict JSON prompt returning KPIs, insights, actions, weather\_outlook, disclaimers
* Local, offline-friendly. Fallback to heuristic JSON if unavailable

### Assumptions

* Office-like occupancy with daytime peaks
* Cooling-focused climate with 24 C setpoint (tunable)
* Common rooms are permissible merge targets
* Sensors are reasonably calibrated; synthetic noise imitates natural variance

### Key Decisions

* Local LLM over cloud to reduce cost and preserve privacy
* Rule-based controls first for determinism; MPC/RL on the roadmap
* CSV and JSONL to keep portability and ease of swapping inputs/outputs
* Free weather API (Open-Meteo) to avoid credentials and cost

---

## 8. Setup

Install Python dependencies:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Note on pyarrow errors: This project does not require pyarrow. If another dependency tries to build it, install a prebuilt wheel:

```bash
pip install "pyarrow==17.0.0" --only-binary=:all:
```

Optional local LLM via Ollama:

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.1:8b
export OLLAMA_MODEL=llama3.1:8b
# Optional: export OLLAMA_BASE_URL=http://localhost:11434
```

Optional MQTT (for actual device integration):

```bash
export MQTT_BROKER=broker.host
export MQTT_PORT=1883
export MQTT_USERNAME=...
export MQTT_PASSWORD=...
export MQTT_TOPIC_PREFIX=building/iot
```

---

## 9. Usage

Generate a 24-hour synthetic dataset at 5-minute resolution:

```bash
python -m src.synth --floors_json data/floors.json --out_dir outputs/synth --hours 24 --step_min 5
```

Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

Dashboard guide:

* Use the sidebar to generate synthetic data at any time
* Use the timeline slider to move through the day
* Tabs

  * Floor Plans: interactive per-floor view with counts at the selected timepoint
  * Trends: total occupancy, energy by floor, and building energy
  * Actions: merge suggestions and derived per-room commands
  * Weather: fetch forecast, view chart and bullets
  * AI Brief: JSON summary via local LLM

---

## 10. Configuration Summary

| Setting             | Location                     | Default                                          | Purpose                       |
| ------------------- | ---------------------------- | ------------------------------------------------ | ----------------------------- |
| FLOORS\_JSON        | streamlit\_app.py / synth.py | data/floors.json                                 | Path to floor plan topology   |
| OLLAMA\_MODEL       | environment variable         | llama3.1:8b                                      | Local LLM name                |
| OLLAMA\_BASE\_URL   | environment variable         | [http://localhost:11434](http://localhost:11434) | Ollama HTTP endpoint          |
| MQTT\_BROKER        | environment variable         | unset                                            | Enables MQTT publish when set |
| MQTT\_PORT          | environment variable         | 1883                                             | MQTT port                     |
| MQTT\_USERNAME      | environment variable         | unset                                            | MQTT auth optional            |
| MQTT\_PASSWORD      | environment variable         | unset                                            | MQTT auth optional            |
| MQTT\_TOPIC\_PREFIX | environment variable         | building/iot                                     | Commands topic namespace      |

---



```

