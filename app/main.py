from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="ecom-cs", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
