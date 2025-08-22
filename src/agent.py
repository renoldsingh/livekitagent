import logging
import aiohttp
import os
import json
import asyncio
import jwt
from datetime import datetime, timedelta
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
    "current_session": None,
    "primary_participant_id": None,
    "participant_metadata": None  # Add this to store metadata
})


def create_jwt_token(participant_id: str) -> str:
    """Create JWT token with participant_id as user_id in payload."""
    try:
        jwt_secret = os.getenv("LIVEKIT_API_JWT")
        if not jwt_secret:
            logger.error("JWT_SECRET not found in environment variables")
            return None
        
        # Current time
        now = datetime.utcnow()
        
        # Token payload with 1 minute expiration
        payload = {
            "user_id": participant_id,
            "iat": now,  # issued at
            "exp": now + timedelta(minutes=1)  # expires in 1 minute
        }
        
        # Create JWT token
        token = jwt.encode(payload, jwt_secret, algorithm="HS256")
        
        logger.info(f"Created JWT token for participant: {participant_id}")
        return token
        
    except Exception as e:
        logger.error(f"Error creating JWT token: {str(e)}")
        return None


class Assistant(Agent):
    def __init__(self) -> None:
        # Get instructions from environment variable
        instructions = os.getenv("ASSISTANT_INSTRUCTIONS", 
            """You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation.
            You are curious, friendly, and have a sense of humor.""")
        
        super().__init__(instructions=instructions)

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
    """Send participant-job mapping to server for tracking with JWT authentication."""
    try:
        server_url = os.getenv("PARTICIPANT_JOB_SERVER_URL")
        if not server_url:
            logger.warning("PARTICIPANT_JOB_SERVER_URL not configured, skipping participant mapping upload")
            return False
        
        participant_id = participant_data.get("participant_id")
        if not participant_id:
            logger.error("No participant_id found in participant_data")
            return False
        
        # Create JWT token
        jwt_token = create_jwt_token(participant_id)
        if not jwt_token:
            logger.error("Failed to create JWT token, skipping API call")
            return False
        
        participant_data["timestamp"] = datetime.utcnow().isoformat()

        logger.info(f"Sending participant-job mapping with JWT: {participant_data}") 
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                server_url,
                json=participant_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info(f"Participant-job mapping sent successfully: {participant_data['participant_id']} -> {participant_data['job_id']}")
                    return True
                else:
                    logger.warning(f"Failed to send participant-job mapping: {response.status}")
                    # Log response body for debugging
                    try:
                        response_text = await response.text()
                        logger.warning(f"Response body: {response_text}")
                    except:
                        pass
                    return False
                    
    except Exception as e:
        logger.error(f"Error sending participant-job mapping: {str(e)}")
        return False


async def send_job_usage_to_server(job_usage: dict, job_id: str):
    """Send job usage data to server with JWT authentication and flush data if successful."""
    try:
        server_url = os.getenv("JOB_USAGE_SERVER_URL")
        if not server_url:
            logger.warning("JOB_USAGE_SERVER_URL not configured, skipping job usage upload")
            return False
        
        # Get participant_id from multiple sources
        participant_id = None
        
        # First try to get from participant_info
        if "participant_info" in job_usage and job_usage["participant_info"]:
            participant_id = job_usage["participant_info"].get("participant_id")
        
        # If not found, try from primary_participant_id
        if not participant_id:
            participant_id = job_usage.get("primary_participant_id")
        
        # If still not found, try from job_usage_data global store
        if not participant_id and job_id in job_usage_data:
            participant_id = job_usage_data[job_id].get("primary_participant_id")
            # Also try participant_info in global store
            if not participant_id and job_usage_data[job_id].get("participant_info"):
                participant_id = job_usage_data[job_id]["participant_info"].get("participant_id")
        
        if not participant_id:
            logger.error(f"No participant_id found in job usage data for job {job_id}")
            logger.error(f"Job usage data keys: {list(job_usage.keys())}")
            logger.error(f"Participant info: {job_usage.get('participant_info', {})}")
            return False
        
        # Create JWT token
        jwt_token = create_jwt_token(participant_id)
        if not jwt_token:
            logger.error("Failed to create JWT token, skipping API call")
            return False
        
        job_usage["timestamp"] = datetime.utcnow().isoformat()
        
        logger.info(f"Sending job usage data with JWT for participant {participant_id}: {job_usage}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                server_url,
                json=job_usage,
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info(f"Job usage data sent successfully for job: {job_usage.get('job_id')}")
                    
                    # Flush data if status is 200
                    if job_id in job_usage_data:
                        logger.info(f"Flushing job usage data for job: {job_id}")
                        del job_usage_data[job_id]
                    
                    return True
                else:
                    logger.warning(f"Failed to send job usage data: {response.status}")
                    # Log response body for debugging
                    try:
                        response_text = await response.text()
                        logger.warning(f"Response body: {response_text}")
                    except:
                        pass
                    return False
                    
    except Exception as e:
        logger.error(f"Error sending job usage data: {str(e)}")
        return False


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

    # Get models from environment variables
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    stt_model = os.getenv("STT_MODEL", "nova-3")
    tts_voice = os.getenv("TTS_VOICE", "6f84f4b8-58a2-430c-8c79-688dad597532")

    # Set up a voice AI pipeline with environment variables
    session = AgentSession(
        llm=openai.LLM(model=llm_model),
        stt=deepgram.STT(model=stt_model, language="multi"),
        tts=cartesia.TTS(voice=tts_voice),
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
        logger.info(f"Participant metadata: {participant.metadata}")
        
        # Map participant to job
        participant_job_mapping[participant.identity] = job_id
        
        # Set primary participant ID and metadata for this job
        job_usage_data[job_id]["primary_participant_id"] = participant.identity
        job_usage_data[job_id]["participant_metadata"] = participant.metadata
        
        # Get session details
        session_details = {
            "llm_model": llm_model,
            "stt_model": stt_model,
            "tts_voice": tts_voice,
        }
        
        logger.info(f"Session details for participant {participant.identity}: {session_details}")
        
        # Store comprehensive participant info in job usage data
        job_usage_data[job_id]["participant_info"] = {
            "participant_id": participant.identity,
            "participant_name": participant.name,
            "participant_metadata": participant.metadata,
            "participant_kind": str(participant.kind) if hasattr(participant, 'kind') else None,
            "join_time": datetime.utcnow().isoformat(),
            "session_details": session_details
        }
        
        # Send participant-job mapping to server
        mapping_data = {
            "event_type": "PARTICIPANT_CONNECTED",
            "participant_id": participant.identity,
            "participant_name": participant.name,
            "job_id": job_id,
            "room_id": ctx.room.name,
            "join_time": datetime.utcnow().isoformat(),
            "participant_metadata": participant.metadata,
            "session_details": session_details  # Add session details here too
        }
            
        # Send and flush if successful
        success = await send_participant_job_mapping(mapping_data)

        # Clean up the detailed data after sending, but keep essential info including participant_id and metadata
        if success:
            # Keep essential info for disconnect tracking and JWT token generation
            job_usage_data[job_id]["participant_info"] = {
                "participant_id": participant.identity,
                "participant_name": participant.name,
                "participant_metadata": participant.metadata,
                "join_time": datetime.utcnow().isoformat()
            }
            logger.info(f"Flushed detailed participant data for {participant.identity}")


    async def handle_participant_disconnected(participant, reason=None):
        logger.info(f"PARTICIPANT DISCONNECTED EVENT: {participant.identity}")
        
        # Get disconnect reason
        disconnect_reason = reason if reason else "unknown"
        if hasattr(participant, 'disconnect_reason'):
            disconnect_reason = participant.disconnect_reason
        elif hasattr(participant, 'metadata') and participant.metadata:
            # Try to extract reason from metadata if available
            try:
                metadata = json.loads(participant.metadata) if isinstance(participant.metadata, str) else participant.metadata
                disconnect_reason = metadata.get('disconnect_reason', disconnect_reason)
            except:
                pass
        
        logger.info(f"Disconnect reason for {participant.identity}: {disconnect_reason}")
        
        # Update leave time if this participant was tracked
        if job_id in job_usage_data and "participant_info" in job_usage_data[job_id]:
            if job_usage_data[job_id]["participant_info"].get("participant_id") == participant.identity:
                job_usage_data[job_id]["participant_info"]["leave_time"] = datetime.utcnow().isoformat()
                job_usage_data[job_id]["participant_info"]["disconnect_reason"] = disconnect_reason
                
                # Ensure metadata is still preserved
                if not job_usage_data[job_id]["participant_info"].get("participant_metadata"):
                    job_usage_data[job_id]["participant_info"]["participant_metadata"] = participant.metadata
                
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
            "event_type": "PARTICIPANT_DISCONNECTED",
            "participant_id": participant.identity,
            "participant_name": participant.name,
            "participant_metadata": participant.metadata,
            "job_id": participant_job_mapping.get(participant.identity, job_id),
            "room_id": ctx.room.name,
            "disconnect_time": datetime.utcnow().isoformat(),
            "disconnect_reason": disconnect_reason
        }
            
        # Send and flush if successful
        success = await send_participant_job_mapping(disconnect_data)
        
        # Clean up participant mapping but keep participant data including metadata for final usage report
        if success:
            # Remove from participant mapping
            participant_job_mapping.pop(participant.identity, None)
            
            # DON'T clear participant info from job data yet - keep it for final usage report including metadata
            logger.info(f"Participant {participant.identity} disconnected but keeping data including metadata for final usage report")

    # Event handlers with reason parameter for disconnection
    @ctx.room.on("participant_connected")
    def _on_participant_connected(participant):
        logger.info(f"Event: participant_connected triggered for {participant.identity}")
        asyncio.create_task(handle_participant_connected(participant))

    @ctx.room.on("participant_disconnected") 
    def _on_participant_disconnected(participant, reason=None):
        logger.info(f"Event: participant_disconnected triggered for {participant.identity}")
        asyncio.create_task(handle_participant_disconnected(participant, reason))

    # Also try alternative event names
    @ctx.room.on("remote_participant_connected")
    def _on_remote_participant_connected(participant):
        logger.info(f"Event: remote_participant_connected triggered for {participant.identity}")
        asyncio.create_task(handle_participant_connected(participant))

    @ctx.room.on("remote_participant_disconnected")
    def _on_remote_participant_disconnected(participant, reason=None):
        logger.info(f"Event: remote_participant_disconnected triggered for {participant.identity}")
        asyncio.create_task(handle_participant_disconnected(participant, reason))

    # Also try participant_joined/left
    @ctx.room.on("participant_joined")
    def _on_participant_joined(participant):
        logger.info(f"Event: participant_joined triggered for {participant.identity}")
        asyncio.create_task(handle_participant_connected(participant))

    @ctx.room.on("participant_left")
    def _on_participant_left(participant, reason=None):
        logger.info(f"Event: participant_left triggered for {participant.identity}")
        asyncio.create_task(handle_participant_disconnected(participant, reason))

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

        # Ensure we have participant info and metadata for final usage report
        if not job_usage_data[job_id].get("primary_participant_id"):
            logger.warning(f"No primary_participant_id found for job {job_id}, trying to extract from other sources")
            # Try to get from participant_info
            if job_usage_data[job_id].get("participant_info", {}).get("participant_id"):
                job_usage_data[job_id]["primary_participant_id"] = job_usage_data[job_id]["participant_info"]["participant_id"]
        
        # Ensure metadata is included at the top level for usage report
        if not job_usage_data[job_id].get("participant_metadata"):
            # Try to get from participant_info
            if job_usage_data[job_id].get("participant_info", {}).get("participant_metadata"):
                job_usage_data[job_id]["participant_metadata"] = job_usage_data[job_id]["participant_info"]["participant_metadata"]

        # Log what metadata we're sending
        metadata_to_send = job_usage_data[job_id].get("participant_metadata")
        logger.info(f"Sending usage data with participant metadata: {metadata_to_send}")

        # Send final job usage to server and flush if successful
        final_job_usage = {
            "event_type": "PARTICIPANT_USAGE",
            **job_usage_data[job_id]
        }
        
        await send_job_usage_to_server(final_job_usage, job_id)

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