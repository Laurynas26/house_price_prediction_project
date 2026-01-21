# AWS Lambda Inference Entry Point

This directory contains the AWS Lambda entry point used to serve model inference behind API Gateway.

The Lambda layer is intentionally thin and delegates all ML logic, schema enforcement, and preprocessing behavior to the shared `PipelineManager`.

---

## Responsibilities

The Lambda handler is responsible for:
- Parsing API Gateway requests
- Managing cold-start initialization
- Delegating inference to `PipelineManager`
- Returning HTTP responses

It does not perform feature engineering, schema enforcement, or model logic.

---

## Cold Start Strategy

A singleton `PipelineManager` is lazily initialized on the first request and reused across invocations within the same execution environment.

This avoids long cold-start times while keeping initialization idempotent and predictable.

---

## Configuration & Runtime Notes

Configuration files are packaged under `/config` and loaded from `LAMBDA_TASK_ROOT`.

Some geolocation and amenities metadata are injected at the Lambda level to accommodate Lambda-specific runtime constraints. Core preprocessing and schema guarantees remain enforced by `PipelineManager`.

---

## Request Contract

The handler supports:
- URL-based inference (`url`)
- Manual feature input (`manual_input`)

Exactly one must be provided per request.

---

## Architectural Boundary

This module is a deployment boundary, not an ML boundary.

All training–inference parity, feature alignment, and model expectations are enforced centrally by `PipelineManager`, allowing the same pipeline to be reused across Lambda, local, and batch inference contexts.
