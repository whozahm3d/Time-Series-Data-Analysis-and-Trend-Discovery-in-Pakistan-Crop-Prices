# 🌾 Pakistan Agricultural Price Intelligence Agent

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Google ADK](https://img.shields.io/badge/Google%20ADK-2.3.0-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/adk-docs/)
[![Gemini](https://img.shields.io/badge/Gemini-flash--lite--latest-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![MCP](https://img.shields.io/badge/MCP-FastMCP-1a1a1a?style=for-the-badge)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Kaggle%20Capstone-blueviolet?style=for-the-badge)](https://github.com/whozahm3d/agri-price-agent)

> A multi-agent system that turns 7.99 million raw crop-price records from Pakistan into a plain-language BUY / SELL / HOLD recommendation for farmers — built with Google's Agent Development Kit, backed by a custom MCP data server, and orchestrated across four specialist agents.
>
> Built for the **5-Day AI Agents: Intensive Vibe Coding Capstone** (Google-sponsored, Kaggle).

---

## Table of Contents

- [Overview](#overview)
- [Why Agents](#why-agents)
- [Architecture](#architecture)
- [Agent Responsibilities](#agent-responsibilities)
- [MCP Server](#mcp-server)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Example Interaction](#example-interaction)
- [Models and Cost Management](#models-and-cost-management)
- [Data Foundation: The Time-Series Analysis Project](#data-foundation-the-time-series-analysis-project)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [Team](#team)
- [License](#license)

---

## Overview

Pakistan's agricultural sector employs nearly half the country's labour force, yet smallholder farmers routinely sell into volatile markets with no access to forward pricing information. This project addresses that gap with an agentic system that:

1. Answers questions about historical crop prices across 138 cities and 76 crops.
2. Forecasts prices for a requested horizon (1–24 months) with confidence bounds.
3. Converts that forecast into a concrete action — buy, sell, hold, or wait — with reasoning a non-technical user can follow.

Rather than a single LLM call over the dataset, the system is decomposed into four cooperating agents, each with a narrow responsibility, coordinated by an orchestrator that enforces a strict "call each step once" execution policy.

## Why Agents

A single-agent, single-prompt design was considered and rejected for this problem:

- **Separation of concerns.** Data retrieval, statistical forecasting, and advisory reasoning are different jobs with different failure modes. Keeping them in separate agents means a bug in the recommendation logic can't corrupt the underlying price data, and vice versa.
- **Tool surface control.** The data agent needs read access to a 400 MB dataset; the forecasting and recommendation agents never touch the raw data at all — they only see the aggregated output of the previous step. This keeps each agent's tool surface minimal and auditable.
- **Composability.** Because `forecast_prices()` and `generate_recommendation()` are plain functions with typed inputs, they can be tested independently of the LLM (see `smoke_test.py`) and reused outside the agent runtime entirely.
- **Traceability.** ADK's event stream lets you see exactly which agent called which tool with which arguments at every step — essential for debugging a pipeline that spans a 7.99M-row dataset.

## Architecture

```
                              ┌──────────────────────────────┐
                              │           USER QUERY          │
                              │  "Should I sell wheat in      │
                              │   Lahore, and what's the      │
                              │   6-month price outlook?"     │
                              └───────────────┬───────────────┘
                                              ▼
                          ┌──────────────────────────────────────┐
                          │        ORCHESTRATOR AGENT             │
                          │  gemini-flash-lite-latest              │
                          │  • No tools of its own                │
                          │  • Delegates via transfer_to_agent    │
                          │  • Enforces: each sub-agent called    │
                          │    at most once per user request      │
                          └───┬───────────────┬──────────────┬────┘
                              │ Step 1         │ Step 2       │ Step 3
                              ▼               ▼              ▼
                  ┌───────────────────┐ ┌──────────────┐ ┌─────────────────────┐
                  │    DATA AGENT      │ │ FORECASTING  │ │  RECOMMENDATION      │
                  │                    │ │    AGENT     │ │      AGENT           │
                  │ MCP tools (via     │ │              │ │                      │
                  │ FastMCP server):   │ │ forecast_    │ │ generate_            │
                  │  get_available_    │ │  prices()    │ │  recommendation()    │
                  │   crops/cities     │ │ summarise_   │ │ generate_market_     │
                  │  get_crop_price_   │ │  forecast()  │ │  strategy()          │
                  │   stats            │ │              │ │                      │
                  │  get_monthly_      │ │ Linear       │ │ Trend + confidence   │
                  │   averages         │ │ regression,  │ │ → BUY/SELL/HOLD,     │
                  │                    │ │ 95% CI       │ │ urgency, risk level  │
                  │ Direct tools:      │ │              │ │                      │
                  │  get_price_history │ │              │ │                      │
                  │  compare_crops_    │ │              │ │                      │
                  │   by_city          │ │              │ │                      │
                  └─────────┬──────────┘ └──────┬───────┘ └──────────┬───────────┘
                            │                    │                   │
                            ▼                    ▼                   ▼
                  ┌───────────────────────────────────────────────────────┐
                  │        cleaned_merged_crop_prices.csv (400 MB)          │
                  │        7.99M records · 138 cities · 76 crops            │
                  │        2008–2024 · produced by the DM project below     │
                  └───────────────────────────────────────────────────────┘
                                              │
                                              ▼
                          ┌──────────────────────────────────────┐
                          │      FINAL FARMER RECOMMENDATION      │
                          │  ACTION (bold, capitalised)           │
                          │  Reasoning in plain language           │
                          │  Confidence + risk level               │
                          │  Forecast table with 95% bounds        │
                          └──────────────────────────────────────┘
```

The orchestrator never calls a sub-agent twice for the same piece of information within a single request — this was a deliberate fix for an early infinite-transfer-loop bug between the orchestrator and the data agent, and is now encoded directly into the orchestrator's instruction as a hard rule.

## Agent Responsibilities

| Agent | Model | Tools | Responsibility |
|---|---|---|---|
| **orchestrator** | `gemini-flash-lite-latest` | none (delegates only) | Routes user queries to the correct sub-agent sequence; tracks step completion to avoid redundant calls; formats the final response |
| **data_agent** | `gemini-flash-lite-latest` | `get_available_crops`, `get_available_cities`, `get_crop_price_stats`, `get_monthly_averages` (via MCP), plus `get_price_history` and `compare_crops_by_city` (direct Python) | Sole point of contact with the dataset; returns structured price data and monthly aggregates |
| **forecasting_agent** | `gemini-flash-lite-latest` | `forecast_prices`, `summarise_forecast` | Fits a linear regression over monthly averages, returns per-month forecasts with 95% prediction intervals, trend direction, and R² |
| **recommendation_agent** | `gemini-flash-lite-latest` | `generate_recommendation`, `generate_market_strategy` | Converts forecast + confidence into an action (BUY/SELL/HOLD/WAIT), urgency, and risk level; can consolidate multiple crop recommendations into a market-wide strategy |

All four agents run on the same model family (`gemini-flash-lite-latest`) to stay within Gemini's free-tier request quota (~1,500 requests/day) while avoiding the aggressive rate limiting seen on `gemini-2.0-flash` and `gemini-2.5-flash` (20 or fewer requests/day on the free tier).

## MCP Server

`mcp_server/server.py` exposes the dataset as a standalone [Model Context Protocol](https://modelcontextprotocol.io/) server built with **FastMCP**, running over stdio. This means any MCP-compatible client — not just this ADK agent — can query the dataset: Claude Desktop, another agent framework, or a CLI tool.

Exposed tools:

| Tool | Purpose |
|---|---|
| `get_available_crops()` | Lists all 76 unique crop names in the dataset |
| `get_available_cities()` | Lists all 138 unique city/market names |
| `get_crop_price_stats(crop_name, city_name)` | Mean, min, max, and latest price for a crop, optionally scoped to a city |
| `get_monthly_averages(crop_name, city_name)` | Monthly average prices — the primary input to the forecasting agent |

The `data_agent` connects to this server via ADK's `MCPToolset` with `StdioServerParameters`, spawning the server as a subprocess and communicating over stdio. Two additional tools (`get_price_history`, `compare_crops_by_city`) are kept as direct in-process Python functions on the data agent rather than MCP tools, since an earlier iteration exposed all six tools through both paths simultaneously and triggered a "duplicate function declaration" error in ADK — the fix was to pick one registration path per tool and remove the overlap.

## Project Structure

```
agri-price-agent/
├── agents/
│   ├── orchestrator/
│   │   └── agent.py              # Root agent, sub-agent routing logic
│   ├── data_agent/
│   │   └── agent.py               # MCP toolset + direct data tools
│   ├── forecasting_agent/
│   │   └── agent.py               # Linear regression forecasting
│   └── recommendation_agent/
│       └── agent.py               # BUY/SELL/HOLD decision logic
├── mcp_server/
│   └── server.py                  # FastMCP server exposing dataset tools
├── data/
│   ├── README.txt                 # Dataset provenance and schema
│   └── cleaned_merged_crop_prices.csv   # Not committed — see Getting Started
├── notebooks/                     # Original Data Mining course notebooks
├── src/                           # Data Mining pipeline modules (preprocessing, modeling, clustering)
├── results/                       # EDA figures, model comparison outputs, forecasts
├── reports/                       # Data Mining project reports (PDF)
├── main.py                        # CLI entry point / programmatic runner
├── smoke_test.py                  # End-to-end test of the agent tool functions (no LLM calls)
├── requirements.txt
├── Makefile                       # Setup, lint, and pipeline targets for the DM project
└── LICENSE
```

## Getting Started

### Prerequisites

- Python 3.12+
- A Gemini API key ([ai.google.dev](https://ai.google.dev/))
- `pip`

### 1. Clone and install

```bash
git clone https://github.com/whozahm3d/agri-price-agent.git
cd agri-price-agent
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Each agent subfolder can optionally define its own `.env` (used here during development to rotate keys across teammates when the free-tier quota was exhausted on one key). A single root-level `.env` is sufficient for a normal run.

### 3. Provide the dataset

The full dataset (`cleaned_merged_crop_prices.csv`, ~400 MB) is not committed to the repository due to size. To regenerate it:

1. Download the raw dataset from Kaggle: [Crop Prices Dataset of Pakistan](https://www.kaggle.com/datasets/humairarana/crop-prices-dataset-of-pakistan) (53 CSV files).
2. Run the Data Mining pipeline in `notebooks/DM_Project_Final_Deliverable.ipynb` to produce the cleaned, merged CSV (see [Data Foundation](#data-foundation-the-time-series-analysis-project) below for what this pipeline does).
3. Place the resulting file at `data/cleaned_merged_crop_prices.csv`.

### 4. Verify the tool layer (optional, no LLM calls)

```bash
python smoke_test.py
```

This runs the full data → forecast → recommendation chain directly against the Python functions, bypassing the LLM entirely — useful for confirming the dataset and tool logic work before spending API quota.

## Usage

**Interactive web UI (recommended):**

```bash
adk web agents/orchestrator
```

Opens the ADK Dev UI in your browser, where you can chat with the orchestrator and inspect every tool call, transfer, and event in real time.

**Programmatic / CLI:**

```bash
python main.py
```

Runs an interactive CLI session, or a single smoke-test query if run non-interactively.

## Example Interaction

```
You: What will wheat cost in Lahore over the next 6 months, and should I sell now?

Agent:
📊 Wheat — Lahore
Current average price: PKR 3,450/40kg (as of latest record)

📈 6-Month Forecast (Linear Regression, R² = 0.82 — high confidence)
  2026-08: PKR 3,510  [3,320 – 3,700]
  2026-09: PKR 3,565  [3,340 – 3,790]
  2026-10: PKR 3,620  [3,355 – 3,885]
  ...

Trend: RISING (~1.6%/month)

ACTION: HOLD / SELL LATER
Risk Level: LOW
Reasoning: Prices are trending upward with high model confidence. Holding
stock for 2–3 months is likely to yield a better price than selling today.
Please confirm final decisions with your local market committee.
```

## Models and Cost Management

All four agents use `gemini-flash-lite-latest`. This choice was made deliberately after testing:

| Model | Free-tier daily quota | Verdict |
|---|---|---|
| `gemini-2.0-flash`, `gemini-2.5-flash` | ≤ 20 requests/day | Too restrictive for a multi-agent pipeline (a single user query can involve 3–4 model calls) |
| `gemini-2.0-flash-lite`, `gemini-1.5-flash` | — | Deprecated / unsupported |
| `gemini-2.5-flash-lite`, `gemini-flash-lite-latest` | ~1,500 requests/day | Selected — sufficient headroom for development, testing, and demo without hitting quota mid-session |

During development, `gemini-flash-lite-latest` was also used to route around intermittent 503 server-congestion errors on pinned model versions, since the `-latest` alias is served from a broader capacity pool.

## Data Foundation: The Time-Series Analysis Project

The 400 MB dataset queried by this agent system is not a raw download — it is the cleaned, merged output of an earlier three-phase Data Mining course project by the same team, applied to Kaggle's [Crop Prices Dataset of Pakistan](https://www.kaggle.com/datasets/humairarana/crop-prices-dataset-of-pakistan) (53 raw CSV files, ~7.99M records, 138 cities, 76 crops, 2008–2024).

That pipeline (`notebooks/`, `src/`) performed:

- **Phase 1 — Cleaning & EDA:** schema validation, non-positive price removal, IQR-based winsorisation (capped, not dropped, to preserve series continuity), calendar feature extraction, 11 exploratory visualisations, STL decomposition, and ADF stationarity testing.
- **Phase 2 — Modeling:** nine forecasting models (naive baselines, ARIMA, Holt-Winters, Linear Regression, Random Forest, XGBoost — default and tuned) evaluated under strict no-leakage, chronologically-split conditions, plus a global-vs-local XGBoost comparison.
- **Phase 3 — Clustering & Anomaly Detection:** K-Means and hierarchical clustering of crop-city pairs by price behaviour, PCA visualisation, and three-method anomaly detection (Z-score, rolling deviation, IQR), linked back to forecasting difficulty.

This work is what makes `cleaned_merged_crop_prices.csv` analysis-ready for the agent system — the forecasting agent's linear regression operates on monthly aggregates that are already outlier-controlled and gap-filtered. Full methodology, figures, and results are in `results/`, `reports/`, and the original notebooks.

## Known Limitations

- **Forecasting model is intentionally simple.** `forecast_prices()` uses ordinary least squares on monthly averages with no dependency on scikit-learn, chosen for speed and zero extra runtime dependencies inside the agent tool layer. It does not use the more sophisticated ARIMA/XGBoost models built in the DM project phase — that remains a natural next step (see [Roadmap](#roadmap)).
- **No input validation / guardrails yet** on user-supplied crop or city names beyond substring matching — malformed or adversarial input is not explicitly sanitised before reaching the dataset query layer.
- **Redundant tool calls.** The `data_agent` occasionally issues the same MCP tool call two to three times within a single turn before returning a result; this does not affect correctness but adds latency and quota usage.
- **Dataset is static.** Prices are not live-updated; forecasts are only as current as the underlying CSV snapshot.

## Roadmap

- [ ] Swap the forecasting agent's OLS model for the tuned XGBoost/ARIMA models already validated in the DM project, exposed as an additional MCP tool.
- [ ] Add input validation and basic prompt-injection guardrails at the orchestrator and data_agent boundary.
- [ ] Reduce redundant tool calls in `data_agent` through explicit result caching per conversation turn.
- [ ] Multi-crop, multi-city batch recommendations via `generate_market_strategy()` exposed directly in the CLI.

## Tech Stack

- **Agent framework:** Google Agent Development Kit (ADK) 2.3.0
- **LLM:** Gemini (`gemini-flash-lite-latest`)
- **Tool protocol:** Model Context Protocol (MCP) via FastMCP
- **Data processing:** pandas, NumPy
- **Data foundation pipeline:** scikit-learn, XGBoost, statsmodels (ARIMA, Holt-Winters), seaborn/matplotlib
- **Language/runtime:** Python 3.12+

## Team

Built for FAST NUCES by:

- **Ali Ahmad** (23L-2619)
- **Taha Nawaz** (23L-2644)
- **Shahzeb Imran** (23L-2506)

The dataset foundation (`notebooks/`, `src/`, `results/`, `reports/`) is joint work from the team's Data Mining course project. The agent system built on top of it for this capstone is documented above.

## License

Released under the [MIT License](LICENSE).
