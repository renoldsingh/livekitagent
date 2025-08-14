import logging
import aiohttp
import os
import json
import asyncio
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Global storage for participant-job mapping and usage tracking
participant_job_mapping = {}
job_usage_data = defaultdict(lambda: {
    "participant_info": {},
    "job_start_time": None,
    "usage_metrics": [],
    "current_session": None
})


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


async def send_participant_job_mapping(participant_data: dict):
    """Send participant-job mapping to server for tracking."""
    try:
        # server_url = os.getenv("PARTICIPANT_JOB_SERVER_URL")
        # if not server_url:
        #     logger.warning("PARTICIPANT_JOB_SERVER_URL not configured, skipping participant mapping upload")
        #     return
        
        participant_data["timestamp"] = datetime.utcnow().isoformat()

        logger.error(f"Sending participant-job mapping: {participant_data}") 
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         server_url,
        #         json=participant_data,
        #         headers={"Content-Type": "application/json"}
        #     ) as response:
        #         if response.status == 200:
        #             logger.info(f"Participant-job mapping sent successfully: {participant_data['participant_id']} -> {participant_data['job_id']}")
        #         else:
        #             logger.warning(f"Failed to send participant-job mapping: {response.status}")
                    
    except Exception as e:
        logger.error(f"Error sending participant-job mapping: {str(e)}")


async def send_job_usage_to_server(job_usage: dict):
    """Send job usage data to server."""
    try:
        # server_url = os.getenv("JOB_USAGE_SERVER_URL")
        # if not server_url:
        #     logger.warning("JOB_USAGE_SERVER_URL not configured, skipping job usage upload")
        #     return
        
        job_usage["timestamp"] = datetime.utcnow().isoformat()
        
        logger.error(f"Sending job usage data: {job_usage}")
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         server_url,
        #         json=job_usage,
        #         headers={"Content-Type": "application/json"}
        #     ) as response:
        #         if response.status == 200:
        #             logger.info(f"Job usage data sent successfully for job: {job_usage.get('job_id')}")
        #         else:
        #             logger.warning(f"Failed to send job usage data: {response.status}")
                    
    except Exception as e:
        logger.error(f"Error sending job usage data: {str(e)}")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    global participant_job_mapping, job_usage_data
    
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Get job ID
    job_id = ctx.job.id if hasattr(ctx, 'job') and hasattr(ctx.job, 'id') else f"job_{ctx.room.name}_{int(datetime.utcnow().timestamp())}"
    
    logger.info(f"Starting job {job_id} in room {ctx.room.name}")
    
    # Initialize job usage data
    job_usage_data[job_id]["job_start_time"] = datetime.utcnow().isoformat()
    job_usage_data[job_id]["room_id"] = ctx.room.name
    job_usage_data[job_id]["job_id"] = job_id

    # Log initial room state
    logger.info(f"Initial room participants: {len(ctx.room.remote_participants)}")
    for participant_id, participant in ctx.room.remote_participants.items():
        logger.info(f"Existing participant: {participant_id} - {participant.identity}")

    # Set up a voice AI pipeline
    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        
        # Store metrics for this job
        job_usage_data[job_id]["usage_metrics"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": ev.metrics.__dict__ if hasattr(ev.metrics, '__dict__') else str(ev.metrics)
        })

    # Define async functions for participant events
    async def handle_participant_connected(participant):
        logger.info(f"PARTICIPANT CONNECTED EVENT: {participant.identity} -> Job: {job_id}")
        logger.info(f"Participant details - Name: {participant.name}, Kind: {getattr(participant, 'kind', 'unknown')}")
        
        # Map participant to job
        participant_job_mapping[participant.identity] = job_id
        
        # Get session details
        session_details = {
            "model": getattr(session.llm, 'model', None),
            "stt_model": getattr(session.stt, 'model', None),
            "tts_voice": getattr(session.tts, 'voice', None),
        }
        
        logger.info(f"Session details for participant {participant.identity}: {session_details}")
        
        # Store participant info in job usage data
        job_usage_data[job_id]["participant_info"] = {
            "participant_id": participant.identity,
            "participant_name": participant.name,
            "participant_metadata": participant.metadata,
            "participant_kind": str(participant.kind) if hasattr(participant, 'kind') else None,
            "join_time": datetime.utcnow().isoformat(),
            "session_details": session_details  # Add session details here
        }
        
        # Send participant-job mapping to server
        mapping_data = {
            "event_type": "participant_job_mapping",
            "participant_id": participant.identity,
            "participant_name": participant.name,
            "job_id": job_id,
            "room_id": ctx.room.name,
            "join_time": datetime.utcnow().isoformat(),
            "participant_metadata": participant.metadata,
            "session_details": session_details  # Add session details here too
        }
            
        # Send and flush
        success = await send_participant_job_mapping(mapping_data)

        # Clean up the detailed data after sending, keep only essential info
        if success:
            # Keep minimal info for disconnect tracking
            job_usage_data[job_id]["participant_info"] = {
                "participant_id": participant.identity,
                "join_time": datetime.utcnow().isoformat()
            }
            logger.info(f"Flushed detailed participant data for {participant.identity}")


    async def handle_participant_disconnected(participant):
        logger.info(f"PARTICIPANT DISCONNECTED EVENT: {participant.identity}")
        
        # Update leave time if this participant was tracked
        if job_id in job_usage_data and "participant_info" in job_usage_data[job_id]:
            if job_usage_data[job_id]["participant_info"].get("participant_id") == participant.identity:
                job_usage_data[job_id]["participant_info"]["leave_time"] = datetime.utcnow().isoformat()
                
                # Calculate session duration
                join_time_str = job_usage_data[job_id]["participant_info"].get("join_time")
                leave_time_str = job_usage_data[job_id]["participant_info"]["leave_time"]
                
                if join_time_str and leave_time_str:
                    join_time = datetime.fromisoformat(join_time_str)
                    leave_time = datetime.fromisoformat(leave_time_str)
                    session_duration = (leave_time - join_time).total_seconds()
                    job_usage_data[job_id]["session_duration_seconds"] = session_duration

        # Send disconnect notification
        disconnect_data = {
            "event_type": "participant_disconnected",
            "participant_id": participant.identity,
            "job_id": participant_job_mapping.get(participant.identity, job_id),
            "room_id": ctx.room.name,
            "disconnect_time": datetime.utcnow().isoformat()
        }
            
        # Send and flush
        success = await send_participant_job_mapping(disconnect_data)
        
        # Clean up all participant data after disconnect
        if success:
            # Remove from participant mapping
            participant_job_mapping.pop(participant.identity, None)
            
            # Clear participant info from job data
            if job_id in job_usage_data:
                job_usage_data[job_id]["participant_info"] = {}
            
            logger.info(f"Flushed all data for disconnected participant {participant.identity}")

    # Try multiple event names to catch participant events
    @ctx.room.on("participant_connected")
    def _on_participant_connected(participant):
        logger.info(f"Event: participant_connected triggered for {participant.identity}")
        asyncio.create_task(handle_participant_connected(participant))

    @ctx.room.on("participant_disconnected") 
    def _on_participant_disconnected(participant):
        logger.info(f"Event: participant_disconnected triggered for {participant.identity}")
        asyncio.create_task(handle_participant_disconnected(participant))

    # Also try alternative event names
    @ctx.room.on("remote_participant_connected")
    def _on_remote_participant_connected(participant):
        logger.info(f"Event: remote_participant_connected triggered for {participant.identity}")
        asyncio.create_task(handle_participant_connected(participant))

    @ctx.room.on("remote_participant_disconnected")
    def _on_remote_participant_disconnected(participant):
        logger.info(f"Event: remote_participant_disconnected triggered for {participant.identity}")
        asyncio.create_task(handle_participant_disconnected(participant))

    # Also try participant_joined/left
    @ctx.room.on("participant_joined")
    def _on_participant_joined(participant):
        logger.info(f"Event: participant_joined triggered for {participant.identity}")
        asyncio.create_task(handle_participant_connected(participant))

    @ctx.room.on("participant_left")
    def _on_participant_left(participant):
        logger.info(f"Event: participant_left triggered for {participant.identity}")
        asyncio.create_task(handle_participant_disconnected(participant))

    # Generic room event logging to see what events are available
    def log_room_event(event_name):
        def _log_event(*args, **kwargs):
            logger.info(f"Room event triggered: {event_name} with args: {args}, kwargs: {kwargs}")
        return _log_event

    # Log all possible room events to debug
    possible_events = [
        "connection_state_changed", "data_received", "room_metadata_changed",
        "track_published", "track_unpublished", "track_subscribed", "track_unsubscribed"
    ]
    
    for event in possible_events:
        try:
            ctx.room.on(event, log_room_event(event))
        except:
            pass  # Event might not exist

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Job {job_id} Usage Summary: {summary}")
        
        # Finalize job usage data
        job_usage_data[job_id]["job_end_time"] = datetime.utcnow().isoformat()
        job_usage_data[job_id]["usage_summary"] = summary.__dict__ if hasattr(summary, '__dict__') else str(summary)
        
        # Calculate total job duration
        start_time_str = job_usage_data[job_id]["job_start_time"]
        end_time_str = job_usage_data[job_id]["job_end_time"]
        
        if start_time_str and end_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str)
            job_duration = (end_time - start_time).total_seconds()
            job_usage_data[job_id]["job_duration_seconds"] = job_duration

        # Send final job usage to server
        final_job_usage = {
            "event_type": "job_completed",
            **job_usage_data[job_id]
        }
        
        await send_job_usage_to_server(final_job_usage)

    ctx.add_shutdown_callback(log_usage)

    logger.info("Starting session...")
    
    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    logger.info("Session started, connecting to room...")

    # Join the room and connect to the user
    await ctx.connect()
    
    logger.info("Connected to room, waiting for participants...")
    
    # After connecting, log current participants again
    logger.info(f"Room participants after connect: {len(ctx.room.remote_participants)}")
    for participant_id, participant in ctx.room.remote_participants.items():
        logger.info(f"Current participant: {participant_id} - {participant.identity}")
        # Manually trigger participant connected for existing participants
        asyncio.create_task(handle_participant_connected(participant))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))