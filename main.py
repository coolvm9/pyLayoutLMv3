import logging
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from services.service_one import router as router_one
from services.service_two import router as router_two
import uvicorn

app = FastAPI()

# Create a custom logger
logger = logging.getLogger(__name__)

# Include routers from your services
app.include_router(router_one)
app.include_router(router_two)


@app.get("/", include_in_schema=False)
async def root():
    logger.info('Redirecting to /docs')
    return RedirectResponse(url='/docs')

if __name__ == "__main__":
    logger.info('Starting application')
    uvicorn.run("main:app", host="0.0.0.0", port=5010, reload=True)
