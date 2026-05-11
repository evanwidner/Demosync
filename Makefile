.PHONY: help install install-light dashboard-install dashboard-dev dashboard-typecheck db-apply-schema db-shell daemon worker-smoke clean

PYTHON ?= python3
ENV_FILE ?= .env

# Load DATABASE_URL from .env if present (lets `make db-apply-schema` "just work").
ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
export
endif

help:
	@echo "Targets:"
	@echo "  install              install worker + daemon python deps (full, incl. nerfstudio)"
	@echo "  install-light        install minimal deps for laptop dev (no nerfstudio)"
	@echo "  dashboard-install    npm install in dashboard/"
	@echo "  dashboard-dev        run Next dev server on :3000"
	@echo "  dashboard-typecheck  tsc --noEmit in dashboard/"
	@echo "  db-apply-schema      apply dashboard/db/schema.sql to \$$DATABASE_URL"
	@echo "  db-shell             open psql against \$$DATABASE_URL"
	@echo "  daemon               run orchestration daemon (DEMOSYNC_DISPATCH=local default)"
	@echo "  worker-smoke         check that ns-process-data, colmap, ffmpeg, ns-train, ns-render are on PATH"
	@echo "  clean                remove worker runs/ and dashboard uploads/"

install:
	$(PYTHON) -m pip install -r worker/requirements.txt -r daemon/requirements.txt

install-light:
	$(PYTHON) -m pip install -r worker/requirements-light.txt -r daemon/requirements.txt

dashboard-install:
	cd dashboard && npm install

dashboard-dev:
	cd dashboard && npm run dev

dashboard-typecheck:
	cd dashboard && npm run typecheck

db-apply-schema:
	@test -n "$$DATABASE_URL" || (echo "DATABASE_URL not set. Fill it in .env (Supabase → Settings → Database)" && exit 1)
	psql "$$DATABASE_URL" -f dashboard/db/schema.sql

db-shell:
	@test -n "$$DATABASE_URL" || (echo "DATABASE_URL not set" && exit 1)
	psql "$$DATABASE_URL"

daemon:
	$(PYTHON) -m daemon.main

worker-smoke:
	$(PYTHON) -m worker.pipeline smoke

clean:
	rm -rf worker/runs
	rm -rf dashboard/uploads
