# Auto-generated model exports
# This file was automatically generated to export all models for easy importing

from __future__ import annotations

# Suppress Pydantic warnings about fields shadowing BaseModel attributes
# (e.g. 'schema' field in OpenAI models shadows deprecated BaseModel.schema())
import warnings
warnings.filterwarnings("ignore", message=".*shadows an attribute in parent.*")

# Import all model modules
try:
    from . import active_status
    from . import add_upload_part_request
    from . import admin_api_key
    from . import annotation
    from . import api_key_list
    from . import apply_patch_call_output_status
    from . import apply_patch_call_output_status_param
    from . import apply_patch_call_status
    from . import apply_patch_call_status_param
    from . import apply_patch_create_file_operation
    from . import apply_patch_create_file_operation_param
    from . import apply_patch_delete_file_operation
    from . import apply_patch_delete_file_operation_param
    from . import apply_patch_operation_param
    from . import apply_patch_tool_call
    from . import apply_patch_tool_call_item_param
    from . import apply_patch_tool_call_output
    from . import apply_patch_tool_call_output_item_param
    from . import apply_patch_tool_param
    from . import apply_patch_update_file_operation
    from . import apply_patch_update_file_operation_param
    from . import approximate_location
    from . import assigned_role_details
    from . import assistant_message_item
    from . import assistant_object
    from . import assistant_stream_event
    from . import assistant_supported_models
    from . import assistant_tool
    from . import assistant_tools_code
    from . import assistant_tools_file_search
    from . import assistant_tools_file_search_type_only
    from . import assistant_tools_function
    from . import assistants_api_response_format_option
    from . import assistants_api_tool_choice_option
    from . import assistants_named_tool_choice
    from . import attachment
    from . import attachment_type
    from . import audio_response_format
    from . import audio_transcription
    from . import audit_log
    from . import audit_log_actor
    from . import audit_log_actor_api_key
    from . import audit_log_actor_service_account
    from . import audit_log_actor_session
    from . import audit_log_actor_user
    from . import audit_log_event_type
    from . import auto_chunking_strategy_request_param
    from . import automatic_thread_titling_param
    from . import batch
    from . import batch_error
    from . import batch_file_expiration_after
    from . import batch_request_counts
    from . import batch_request_input
    from . import batch_request_output
    from . import certificate
    from . import chat_completion_allowed_tools
    from . import chat_completion_allowed_tools_choice
    from . import chat_completion_deleted
    from . import chat_completion_function_call_option
    from . import chat_completion_functions
    from . import chat_completion_list
    from . import chat_completion_message_custom_tool_call
    from . import chat_completion_message_list
    from . import chat_completion_message_tool_call
    from . import chat_completion_message_tool_call_chunk
    from . import chat_completion_message_tool_calls
    from . import chat_completion_modalities
    from . import chat_completion_named_tool_choice
    from . import chat_completion_named_tool_choice_custom
    from . import chat_completion_request_assistant_message
    from . import chat_completion_request_assistant_message_content_part
    from . import chat_completion_request_developer_message
    from . import chat_completion_request_function_message
    from . import chat_completion_request_message
    from . import chat_completion_request_message_content_part_audio
    from . import chat_completion_request_message_content_part_file
    from . import chat_completion_request_message_content_part_image
    from . import chat_completion_request_message_content_part_refusal
    from . import chat_completion_request_message_content_part_text
    from . import chat_completion_request_system_message
    from . import chat_completion_request_system_message_content_part
    from . import chat_completion_request_tool_message
    from . import chat_completion_request_tool_message_content_part
    from . import chat_completion_request_user_message
    from . import chat_completion_request_user_message_content_part
    from . import chat_completion_response_message
    from . import chat_completion_role
    from . import chat_completion_stream_options
    from . import chat_completion_stream_response_delta
    from . import chat_completion_token_logprob
    from . import chat_completion_tool
    from . import chat_completion_tool_choice_option
    from . import chat_model
    from . import chat_session_automatic_thread_titling
    from . import chat_session_chatkit_configuration
    from . import chat_session_file_upload
    from . import chat_session_history
    from . import chat_session_rate_limits
    from . import chat_session_resource
    from . import chat_session_status
    from . import chatkit_configuration_param
    from . import chatkit_workflow
    from . import chatkit_workflow_tracing
    from . import chunking_strategy_request_param
    from . import chunking_strategy_response
    from . import click_button_type
    from . import click_param
    from . import client_tool_call_item
    from . import client_tool_call_status
    from . import closed_status
    from . import code_interpreter_container_auto
    from . import code_interpreter_file_output
    from . import code_interpreter_output_image
    from . import code_interpreter_output_logs
    from . import code_interpreter_text_output
    from . import code_interpreter_tool
    from . import code_interpreter_tool_call
    from . import compact_resource
    from . import compact_response_method_public_body
    from . import compaction_body
    from . import compaction_summary_item_param
    from . import comparison_filter
    from . import comparison_filter_value_items
    from . import complete_upload_request
    from . import completion_usage
    from . import compound_filter
    from . import computer_action
    from . import computer_call_output_item_param
    from . import computer_call_safety_check_param
    from . import computer_environment
    from . import computer_screenshot_content
    from . import computer_screenshot_image
    from . import computer_tool_call
    from . import computer_tool_call_output
    from . import computer_tool_call_output_resource
    from . import computer_use_preview_tool
    from . import container_file_citation_body
    from . import container_file_list_resource
    from . import container_file_resource
    from . import container_list_resource
    from . import container_memory_limit
    from . import container_resource
    from . import content
    from . import conversation
    from . import conversation_2
    from . import conversation_item
    from . import conversation_item_list
    from . import conversation_param
    from . import conversation_param_2
    from . import conversation_resource
    from . import costs_result
    from . import create_assistant_request
    from . import create_chat_completion_request
    from . import create_chat_completion_response
    from . import create_chat_completion_stream_response
    from . import create_chat_session_body
    from . import create_completion_request
    from . import create_completion_response
    from . import create_container_body
    from . import create_container_file_body
    from . import create_conversation_body
    from . import create_embedding_request
    from . import create_embedding_response
    from . import create_eval_completions_run_data_source
    from . import create_eval_custom_data_source_config
    from . import create_eval_item
    from . import create_eval_jsonl_run_data_source
    from . import create_eval_label_model_grader
    from . import create_eval_logs_data_source_config
    from . import create_eval_request
    from . import create_eval_responses_run_data_source
    from . import create_eval_run_request
    from . import create_eval_stored_completions_data_source_config
    from . import create_file_request
    from . import create_fine_tuning_checkpoint_permission_request
    from . import create_fine_tuning_job_request
    from . import create_group_body
    from . import create_group_user_body
    from . import create_image_edit_request
    from . import create_image_request
    from . import create_image_variation_request
    from . import create_message_request
    from . import create_model_response_properties
    from . import create_moderation_request
    from . import create_moderation_response
    from . import create_response
    from . import create_run_request
    from . import create_run_request_without_stream
    from . import create_speech_request
    from . import create_speech_response_stream_event
    from . import create_thread_and_run_request
    from . import create_thread_and_run_request_without_stream
    from . import create_thread_request
    from . import create_transcription_request
    from . import create_transcription_response_diarized_json
    from . import create_transcription_response_json
    from . import create_transcription_response_stream_event
    from . import create_transcription_response_verbose_json
    from . import create_translation_request
    from . import create_translation_response_json
    from . import create_translation_response_verbose_json
    from . import create_upload_request
    from . import create_vector_store_file_batch_request
    from . import create_vector_store_file_request
    from . import create_vector_store_request
    from . import create_video_body
    from . import create_video_remix_body
    from . import create_voice_consent_request
    from . import create_voice_request
    from . import custom_grammar_format_param
    from . import custom_text_format_param
    from . import custom_tool_call
    from . import custom_tool_call_output
    from . import custom_tool_chat_completions
    from . import custom_tool_param
    from . import delete_assistant_response
    from . import delete_certificate_response
    from . import delete_file_response
    from . import delete_fine_tuning_checkpoint_permission_response
    from . import delete_message_response
    from . import delete_model_response
    from . import delete_thread_response
    from . import delete_vector_store_file_response
    from . import delete_vector_store_response
    from . import deleted_conversation
    from . import deleted_conversation_resource
    from . import deleted_role_assignment_resource
    from . import deleted_thread_resource
    from . import deleted_video_resource
    from . import detail_enum
    from . import done_event
    from . import double_click_action
    from . import drag
    from . import drag_point
    from . import easy_input_message
    from . import embedding
    from . import error
    from . import error_2
    from . import error_event
    from . import error_response
    from . import eval
    from . import eval_api_error
    from . import eval_custom_data_source_config
    from . import eval_grader_label_model
    from . import eval_grader_python
    from . import eval_grader_score_model
    from . import eval_grader_string_check
    from . import eval_grader_text_similarity
    from . import eval_item
    from . import eval_item_content
    from . import eval_item_content_array
    from . import eval_item_content_item
    from . import eval_item_content_output_text
    from . import eval_item_content_text
    from . import eval_item_input_image
    from . import eval_jsonl_file_content_source
    from . import eval_jsonl_file_id_source
    from . import eval_list
    from . import eval_logs_data_source_config
    from . import eval_responses_source
    from . import eval_run
    from . import eval_run_list
    from . import eval_run_output_item
    from . import eval_run_output_item_list
    from . import eval_run_output_item_result
    from . import eval_stored_completions_data_source_config
    from . import eval_stored_completions_source
    from . import expires_after_param
    from . import file_annotation
    from . import file_annotation_source
    from . import file_citation_body
    from . import file_expiration_after
    from . import file_path
    from . import file_purpose
    from . import file_search_ranker
    from . import file_search_ranking_options
    from . import file_search_tool
    from . import file_search_tool_call
    from . import file_upload_param
    from . import filters
    from . import fine_tune_chat_completion_request_assistant_message
    from . import fine_tune_chat_request_input
    from . import fine_tune_dpo_hyperparameters
    from . import fine_tune_dpo_method
    from . import fine_tune_method
    from . import fine_tune_preference_request_input
    from . import fine_tune_reinforcement_hyperparameters
    from . import fine_tune_reinforcement_method
    from . import fine_tune_reinforcement_request_input
    from . import fine_tune_supervised_hyperparameters
    from . import fine_tune_supervised_method
    from . import fine_tuning_checkpoint_permission
    from . import fine_tuning_integration
    from . import fine_tuning_job
    from . import fine_tuning_job_checkpoint
    from . import fine_tuning_job_event
    from . import function_and_custom_tool_call_output
    from . import function_call_item_status
    from . import function_call_output_item_param
    from . import function_object
    from . import function_parameters
    from . import function_shell_action
    from . import function_shell_action_param
    from . import function_shell_call
    from . import function_shell_call_item_param
    from . import function_shell_call_item_status
    from . import function_shell_call_output
    from . import function_shell_call_output_content
    from . import function_shell_call_output_content_param
    from . import function_shell_call_output_exit_outcome
    from . import function_shell_call_output_exit_outcome_param
    from . import function_shell_call_output_item_param
    from . import function_shell_call_output_outcome_param
    from . import function_shell_call_output_timeout_outcome
    from . import function_shell_call_output_timeout_outcome_param
    from . import function_shell_tool_param
    from . import function_tool
    from . import function_tool_call
    from . import function_tool_call_output
    from . import function_tool_call_output_resource
    from . import function_tool_call_resource
    from . import grader_label_model
    from . import grader_multi
    from . import grader_python
    from . import grader_score_model
    from . import grader_string_check
    from . import grader_text_similarity
    from . import grammar_syntax1
    from . import group
    from . import group_deleted_resource
    from . import group_list_resource
    from . import group_resource_with_success
    from . import group_response
    from . import group_role_assignment
    from . import group_user_assignment
    from . import group_user_deleted_resource
    from . import history_param
    from . import hybrid_search_options
    from . import image
    from . import image_detail
    from . import image_edit_completed_event
    from . import image_edit_partial_image_event
    from . import image_edit_stream_event
    from . import image_gen_completed_event
    from . import image_gen_input_usage_details
    from . import image_gen_output_tokens_details
    from . import image_gen_partial_image_event
    from . import image_gen_stream_event
    from . import image_gen_tool
    from . import image_gen_tool_call
    from . import image_gen_usage
    from . import images_response
    from . import images_usage
    from . import include_enum
    from . import inference_options
    from . import input_audio
    from . import input_content
    from . import input_fidelity
    from . import input_file_content
    from . import input_file_content_param
    from . import input_image_content
    from . import input_image_content_param_auto_param
    from . import input_item
    from . import input_message
    from . import input_message_content_list
    from . import input_message_resource
    from . import input_param
    from . import input_text_content
    from . import input_text_content_param
    from . import invite
    from . import invite_delete_response
    from . import invite_list_response
    from . import invite_project_group_body
    from . import invite_request
    from . import item
    from . import item_field
    from . import item_reference_param
    from . import item_resource
    from . import key_press_action
    from . import list_assistants_response
    from . import list_audit_logs_response
    from . import list_batches_response
    from . import list_certificates_response
    from . import list_files_response
    from . import list_fine_tuning_checkpoint_permission_response
    from . import list_fine_tuning_job_checkpoints_response
    from . import list_fine_tuning_job_events_response
    from . import list_messages_response
    from . import list_models_response
    from . import list_paginated_fine_tuning_jobs_response
    from . import list_run_steps_response
    from . import list_runs_response
    from . import list_vector_store_files_response
    from . import list_vector_stores_response
    from . import local_shell_call_status
    from . import local_shell_exec_action
    from . import local_shell_tool_call
    from . import local_shell_tool_call_output
    from . import local_shell_tool_param
    from . import locked_status
    from . import log_prob
    from . import log_prob_properties
    from . import mcp_approval_request
    from . import mcp_approval_response
    from . import mcp_approval_response_resource
    from . import mcp_list_tools
    from . import mcp_list_tools_tool
    from . import mcp_tool
    from . import mcp_tool_call
    from . import mcp_tool_call_status
    from . import mcp_tool_filter
    from . import message
    from . import message_content
    from . import message_content_delta
    from . import message_content_image_file_object
    from . import message_content_image_url_object
    from . import message_content_refusal_object
    from . import message_content_text_annotations_file_citation_object
    from . import message_content_text_annotations_file_path_object
    from . import message_content_text_object
    from . import message_delta_content_image_file_object
    from . import message_delta_content_image_url_object
    from . import message_delta_content_refusal_object
    from . import message_delta_content_text_annotations_file_citation_object
    from . import message_delta_content_text_annotations_file_path_object
    from . import message_delta_content_text_object
    from . import message_delta_object
    from . import message_object
    from . import message_request_content_text_object
    from . import message_role
    from . import message_status
    from . import message_stream_event
    from . import metadata
    from . import model
    from . import model_ids
    from . import model_ids_compaction
    from . import model_ids_responses
    from . import model_ids_shared
    from . import model_response_properties
    from . import moderation_image_url_input
    from . import moderation_text_input
    from . import modify_assistant_request
    from . import modify_certificate_request
    from . import modify_message_request
    from . import modify_run_request
    from . import modify_thread_request
    from . import move
    from . import noise_reduction_type
    from . import open_ai_file
    from . import order_enum
    from . import other_chunking_strategy_response_param
    from . import output_audio
    from . import output_content
    from . import output_item
    from . import output_message
    from . import output_message_content
    from . import output_text_content
    from . import parallel_tool_calls
    from . import partial_images
    from . import prediction_content
    from . import project
    from . import project_api_key
    from . import project_api_key_delete_response
    from . import project_api_key_list_response
    from . import project_create_request
    from . import project_group
    from . import project_group_deleted_resource
    from . import project_group_list_resource
    from . import project_list_response
    from . import project_rate_limit
    from . import project_rate_limit_list_response
    from . import project_rate_limit_update_request
    from . import project_service_account
    from . import project_service_account_api_key
    from . import project_service_account_create_request
    from . import project_service_account_create_response
    from . import project_service_account_delete_response
    from . import project_service_account_list_response
    from . import project_update_request
    from . import project_user
    from . import project_user_create_request
    from . import project_user_delete_response
    from . import project_user_list_response
    from . import project_user_update_request
    from . import prompt
    from . import public_assign_organization_group_role_body
    from . import public_create_organization_role_body
    from . import public_role_list_resource
    from . import public_update_organization_role_body
    from . import ranker_version_type
    from . import ranking_options
    from . import rate_limits_param
    from . import realtime_audio_formats
    from . import realtime_beta_client_event_conversation_item_create
    from . import realtime_beta_client_event_conversation_item_delete
    from . import realtime_beta_client_event_conversation_item_retrieve
    from . import realtime_beta_client_event_conversation_item_truncate
    from . import realtime_beta_client_event_input_audio_buffer_append
    from . import realtime_beta_client_event_input_audio_buffer_clear
    from . import realtime_beta_client_event_input_audio_buffer_commit
    from . import realtime_beta_client_event_output_audio_buffer_clear
    from . import realtime_beta_client_event_response_cancel
    from . import realtime_beta_client_event_response_create
    from . import realtime_beta_client_event_session_update
    from . import realtime_beta_client_event_transcription_session_update
    from . import realtime_beta_response
    from . import realtime_beta_response_create_params
    from . import realtime_beta_server_event_conversation_item_created
    from . import realtime_beta_server_event_conversation_item_deleted
    from . import realtime_beta_server_event_conversation_item_input_audio_transcription_completed
    from . import realtime_beta_server_event_conversation_item_input_audio_transcription_delta
    from . import realtime_beta_server_event_conversation_item_input_audio_transcription_failed
    from . import realtime_beta_server_event_conversation_item_input_audio_transcription_segment
    from . import realtime_beta_server_event_conversation_item_retrieved
    from . import realtime_beta_server_event_conversation_item_truncated
    from . import realtime_beta_server_event_error
    from . import realtime_beta_server_event_input_audio_buffer_cleared
    from . import realtime_beta_server_event_input_audio_buffer_committed
    from . import realtime_beta_server_event_input_audio_buffer_speech_started
    from . import realtime_beta_server_event_input_audio_buffer_speech_stopped
    from . import realtime_beta_server_event_mcp_list_tools_completed
    from . import realtime_beta_server_event_mcp_list_tools_failed
    from . import realtime_beta_server_event_mcp_list_tools_in_progress
    from . import realtime_beta_server_event_rate_limits_updated
    from . import realtime_beta_server_event_response_audio_delta
    from . import realtime_beta_server_event_response_audio_done
    from . import realtime_beta_server_event_response_audio_transcript_delta
    from . import realtime_beta_server_event_response_audio_transcript_done
    from . import realtime_beta_server_event_response_content_part_added
    from . import realtime_beta_server_event_response_content_part_done
    from . import realtime_beta_server_event_response_created
    from . import realtime_beta_server_event_response_done
    from . import realtime_beta_server_event_response_function_call_arguments_delta
    from . import realtime_beta_server_event_response_function_call_arguments_done
    from . import realtime_beta_server_event_response_mcp_call_arguments_delta
    from . import realtime_beta_server_event_response_mcp_call_arguments_done
    from . import realtime_beta_server_event_response_mcp_call_completed
    from . import realtime_beta_server_event_response_mcp_call_failed
    from . import realtime_beta_server_event_response_mcp_call_in_progress
    from . import realtime_beta_server_event_response_output_item_added
    from . import realtime_beta_server_event_response_output_item_done
    from . import realtime_beta_server_event_response_text_delta
    from . import realtime_beta_server_event_response_text_done
    from . import realtime_beta_server_event_session_created
    from . import realtime_beta_server_event_session_updated
    from . import realtime_beta_server_event_transcription_session_created
    from . import realtime_beta_server_event_transcription_session_updated
    from . import realtime_call_create_request
    from . import realtime_call_refer_request
    from . import realtime_call_reject_request
    from . import realtime_client_event
    from . import realtime_client_event_conversation_item_create
    from . import realtime_client_event_conversation_item_delete
    from . import realtime_client_event_conversation_item_retrieve
    from . import realtime_client_event_conversation_item_truncate
    from . import realtime_client_event_input_audio_buffer_append
    from . import realtime_client_event_input_audio_buffer_clear
    from . import realtime_client_event_input_audio_buffer_commit
    from . import realtime_client_event_output_audio_buffer_clear
    from . import realtime_client_event_response_cancel
    from . import realtime_client_event_response_create
    from . import realtime_client_event_session_update
    from . import realtime_client_event_transcription_session_update
    from . import realtime_connect_params
    from . import realtime_conversation_item
    from . import realtime_conversation_item_function_call
    from . import realtime_conversation_item_function_call_output
    from . import realtime_conversation_item_message_assistant
    from . import realtime_conversation_item_message_system
    from . import realtime_conversation_item_message_user
    from . import realtime_conversation_item_with_reference
    from . import realtime_create_client_secret_request
    from . import realtime_create_client_secret_response
    from . import realtime_function_tool
    from . import realtime_mcp_approval_request
    from . import realtime_mcp_approval_response
    from . import realtime_mcp_list_tools
    from . import realtime_mcp_protocol_error
    from . import realtime_mcp_tool_call
    from . import realtime_mcp_tool_execution_error
    from . import realtime_mcphttp_error
    from . import realtime_response
    from . import realtime_response_create_params
    from . import realtime_server_event
    from . import realtime_server_event_conversation_created
    from . import realtime_server_event_conversation_item_added
    from . import realtime_server_event_conversation_item_created
    from . import realtime_server_event_conversation_item_deleted
    from . import realtime_server_event_conversation_item_done
    from . import realtime_server_event_conversation_item_input_audio_transcription_completed
    from . import realtime_server_event_conversation_item_input_audio_transcription_delta
    from . import realtime_server_event_conversation_item_input_audio_transcription_failed
    from . import realtime_server_event_conversation_item_input_audio_transcription_segment
    from . import realtime_server_event_conversation_item_retrieved
    from . import realtime_server_event_conversation_item_truncated
    from . import realtime_server_event_error
    from . import realtime_server_event_input_audio_buffer_cleared
    from . import realtime_server_event_input_audio_buffer_committed
    from . import realtime_server_event_input_audio_buffer_dtmf_event_received
    from . import realtime_server_event_input_audio_buffer_speech_started
    from . import realtime_server_event_input_audio_buffer_speech_stopped
    from . import realtime_server_event_input_audio_buffer_timeout_triggered
    from . import realtime_server_event_mcp_list_tools_completed
    from . import realtime_server_event_mcp_list_tools_failed
    from . import realtime_server_event_mcp_list_tools_in_progress
    from . import realtime_server_event_output_audio_buffer_cleared
    from . import realtime_server_event_output_audio_buffer_started
    from . import realtime_server_event_output_audio_buffer_stopped
    from . import realtime_server_event_rate_limits_updated
    from . import realtime_server_event_response_audio_delta
    from . import realtime_server_event_response_audio_done
    from . import realtime_server_event_response_audio_transcript_delta
    from . import realtime_server_event_response_audio_transcript_done
    from . import realtime_server_event_response_content_part_added
    from . import realtime_server_event_response_content_part_done
    from . import realtime_server_event_response_created
    from . import realtime_server_event_response_done
    from . import realtime_server_event_response_function_call_arguments_delta
    from . import realtime_server_event_response_function_call_arguments_done
    from . import realtime_server_event_response_mcp_call_arguments_delta
    from . import realtime_server_event_response_mcp_call_arguments_done
    from . import realtime_server_event_response_mcp_call_completed
    from . import realtime_server_event_response_mcp_call_failed
    from . import realtime_server_event_response_mcp_call_in_progress
    from . import realtime_server_event_response_output_item_added
    from . import realtime_server_event_response_output_item_done
    from . import realtime_server_event_response_text_delta
    from . import realtime_server_event_response_text_done
    from . import realtime_server_event_session_created
    from . import realtime_server_event_session_updated
    from . import realtime_server_event_transcription_session_updated
    from . import realtime_session
    from . import realtime_session_create_request
    from . import realtime_session_create_request_ga
    from . import realtime_session_create_response
    from . import realtime_session_create_response_ga
    from . import realtime_transcription_session_create_request
    from . import realtime_transcription_session_create_request_ga
    from . import realtime_transcription_session_create_response
    from . import realtime_transcription_session_create_response_ga
    from . import realtime_truncation
    from . import realtime_turn_detection
    from . import reasoning
    from . import reasoning_effort
    from . import reasoning_item
    from . import reasoning_text_content
    from . import refusal_content
    from . import response
    from . import response_audio_delta_event
    from . import response_audio_done_event
    from . import response_audio_transcript_delta_event
    from . import response_audio_transcript_done_event
    from . import response_code_interpreter_call_code_delta_event
    from . import response_code_interpreter_call_code_done_event
    from . import response_code_interpreter_call_completed_event
    from . import response_code_interpreter_call_in_progress_event
    from . import response_code_interpreter_call_interpreting_event
    from . import response_completed_event
    from . import response_content_part_added_event
    from . import response_content_part_done_event
    from . import response_created_event
    from . import response_custom_tool_call_input_delta_event
    from . import response_custom_tool_call_input_done_event
    from . import response_error
    from . import response_error_code
    from . import response_error_event
    from . import response_failed_event
    from . import response_file_search_call_completed_event
    from . import response_file_search_call_in_progress_event
    from . import response_file_search_call_searching_event
    from . import response_format_json_object
    from . import response_format_json_schema
    from . import response_format_json_schema_schema
    from . import response_format_text
    from . import response_format_text_grammar
    from . import response_format_text_python
    from . import response_function_call_arguments_delta_event
    from . import response_function_call_arguments_done_event
    from . import response_image_gen_call_completed_event
    from . import response_image_gen_call_generating_event
    from . import response_image_gen_call_in_progress_event
    from . import response_image_gen_call_partial_image_event
    from . import response_in_progress_event
    from . import response_incomplete_event
    from . import response_item_list
    from . import response_log_prob
    from . import response_mcp_call_arguments_delta_event
    from . import response_mcp_call_arguments_done_event
    from . import response_mcp_call_completed_event
    from . import response_mcp_call_failed_event
    from . import response_mcp_call_in_progress_event
    from . import response_mcp_list_tools_completed_event
    from . import response_mcp_list_tools_failed_event
    from . import response_mcp_list_tools_in_progress_event
    from . import response_modalities
    from . import response_output_item_added_event
    from . import response_output_item_done_event
    from . import response_output_text
    from . import response_output_text_annotation_added_event
    from . import response_prompt_variables
    from . import response_properties
    from . import response_queued_event
    from . import response_reasoning_summary_part_added_event
    from . import response_reasoning_summary_part_done_event
    from . import response_reasoning_summary_text_delta_event
    from . import response_reasoning_summary_text_done_event
    from . import response_reasoning_text_delta_event
    from . import response_reasoning_text_done_event
    from . import response_refusal_delta_event
    from . import response_refusal_done_event
    from . import response_stream_event
    from . import response_stream_options
    from . import response_text_delta_event
    from . import response_text_done_event
    from . import response_text_param
    from . import response_usage
    from . import response_web_search_call_completed_event
    from . import response_web_search_call_in_progress_event
    from . import response_web_search_call_searching_event
    from . import role
    from . import role_deleted_resource
    from . import role_list_resource
    from . import run_completion_usage
    from . import run_grader_request
    from . import run_grader_response
    from . import run_object
    from . import run_status
    from . import run_step_completion_usage
    from . import run_step_delta_object
    from . import run_step_delta_object_delta
    from . import run_step_delta_step_details_message_creation_object
    from . import run_step_delta_step_details_tool_call
    from . import run_step_delta_step_details_tool_calls_code_object
    from . import run_step_delta_step_details_tool_calls_code_output_image_object
    from . import run_step_delta_step_details_tool_calls_code_output_logs_object
    from . import run_step_delta_step_details_tool_calls_file_search_object
    from . import run_step_delta_step_details_tool_calls_function_object
    from . import run_step_delta_step_details_tool_calls_object
    from . import run_step_details_message_creation_object
    from . import run_step_details_tool_call
    from . import run_step_details_tool_calls_code_object
    from . import run_step_details_tool_calls_code_output_image_object
    from . import run_step_details_tool_calls_code_output_logs_object
    from . import run_step_details_tool_calls_file_search_object
    from . import run_step_details_tool_calls_file_search_ranking_options_object
    from . import run_step_details_tool_calls_file_search_result_object
    from . import run_step_details_tool_calls_function_object
    from . import run_step_details_tool_calls_object
    from . import run_step_object
    from . import run_step_stream_event
    from . import run_stream_event
    from . import run_tool_call_object
    from . import screenshot
    from . import scroll
    from . import search_context_size
    from . import service_tier
    from . import specific_apply_patch_param
    from . import specific_function_shell_param
    from . import speech_audio_delta_event
    from . import speech_audio_done_event
    from . import static_chunking_strategy
    from . import static_chunking_strategy_request_param
    from . import static_chunking_strategy_response_param
    from . import stop_configuration
    from . import submit_tool_outputs_run_request
    from . import submit_tool_outputs_run_request_without_stream
    from . import summary
    from . import summary_text_content
    from . import task_group_item
    from . import task_group_task
    from . import task_item
    from . import task_type
    from . import text_annotation
    from . import text_annotation_delta
    from . import text_content
    from . import text_response_format_configuration
    from . import text_response_format_json_schema
    from . import thread_item
    from . import thread_item_list_resource
    from . import thread_list_resource
    from . import thread_object
    from . import thread_resource
    from . import thread_stream_event
    from . import toggle_certificates_request
    from . import token_counts_body
    from . import token_counts_resource
    from . import tool
    from . import tool_choice
    from . import tool_choice_allowed
    from . import tool_choice_custom
    from . import tool_choice_function
    from . import tool_choice_mcp
    from . import tool_choice_options
    from . import tool_choice_param
    from . import tool_choice_types
    from . import tools_array
    from . import top_log_prob
    from . import transcript_text_delta_event
    from . import transcript_text_done_event
    from . import transcript_text_segment_event
    from . import transcript_text_usage_duration
    from . import transcript_text_usage_tokens
    from . import transcription_chunking_strategy
    from . import transcription_diarized_segment
    from . import transcription_include
    from . import transcription_segment
    from . import transcription_word
    from . import truncation_enum
    from . import truncation_object
    from . import type
    from . import update_conversation_body
    from . import update_group_body
    from . import update_vector_store_file_attributes_request
    from . import update_vector_store_request
    from . import update_voice_consent_request
    from . import upload
    from . import upload_certificate_request
    from . import upload_part
    from . import url_annotation
    from . import url_annotation_source
    from . import url_citation_body
    from . import usage_audio_speeches_result
    from . import usage_audio_transcriptions_result
    from . import usage_code_interpreter_sessions_result
    from . import usage_completions_result
    from . import usage_embeddings_result
    from . import usage_images_result
    from . import usage_moderations_result
    from . import usage_response
    from . import usage_time_bucket
    from . import usage_vector_stores_result
    from . import user
    from . import user_delete_response
    from . import user_list_resource
    from . import user_list_response
    from . import user_message_input_text
    from . import user_message_item
    from . import user_message_quoted_text
    from . import user_role_assignment
    from . import user_role_update_request
    from . import vad_config
    from . import validate_grader_request
    from . import validate_grader_response
    from . import vector_store_expiration_after
    from . import vector_store_file_attributes
    from . import vector_store_file_batch_object
    from . import vector_store_file_content_response
    from . import vector_store_file_object
    from . import vector_store_object
    from . import vector_store_search_request
    from . import vector_store_search_result_content_object
    from . import vector_store_search_result_item
    from . import vector_store_search_results_page
    from . import verbosity
    from . import video_content_variant
    from . import video_list_resource
    from . import video_model
    from . import video_resource
    from . import video_seconds
    from . import video_size
    from . import video_status
    from . import voice_consent_deleted_resource
    from . import voice_consent_list_resource
    from . import voice_consent_resource
    from . import voice_ids_or_custom_voice
    from . import voice_ids_shared
    from . import voice_resource
    from . import wait
    from . import web_search_action_find
    from . import web_search_action_open_page
    from . import web_search_action_search
    from . import web_search_approximate_location
    from . import web_search_context_size
    from . import web_search_location
    from . import web_search_preview_tool
    from . import web_search_tool
    from . import web_search_tool_call
    from . import webhook_batch_cancelled
    from . import webhook_batch_completed
    from . import webhook_batch_expired
    from . import webhook_batch_failed
    from . import webhook_eval_run_canceled
    from . import webhook_eval_run_failed
    from . import webhook_eval_run_succeeded
    from . import webhook_fine_tuning_job_cancelled
    from . import webhook_fine_tuning_job_failed
    from . import webhook_fine_tuning_job_succeeded
    from . import webhook_realtime_call_incoming
    from . import webhook_response_cancelled
    from . import webhook_response_completed
    from . import webhook_response_failed
    from . import webhook_response_incomplete
    from . import widget_message_item
    from . import workflow_param
    from . import workflow_tracing_param
except ImportError as e:
    import sys
    print(f"Warning: Some model modules could not be imported: {e}", file=sys.stderr)

# Define what gets imported with 'from models import *'
__all__ = [
    'active_status',
    'add_upload_part_request',
    'admin_api_key',
    'annotation',
    'api_key_list',
    'apply_patch_call_output_status',
    'apply_patch_call_output_status_param',
    'apply_patch_call_status',
    'apply_patch_call_status_param',
    'apply_patch_create_file_operation',
    'apply_patch_create_file_operation_param',
    'apply_patch_delete_file_operation',
    'apply_patch_delete_file_operation_param',
    'apply_patch_operation_param',
    'apply_patch_tool_call',
    'apply_patch_tool_call_item_param',
    'apply_patch_tool_call_output',
    'apply_patch_tool_call_output_item_param',
    'apply_patch_tool_param',
    'apply_patch_update_file_operation',
    'apply_patch_update_file_operation_param',
    'approximate_location',
    'assigned_role_details',
    'assistant_message_item',
    'assistant_object',
    'assistant_stream_event',
    'assistant_supported_models',
    'assistant_tool',
    'assistant_tools_code',
    'assistant_tools_file_search',
    'assistant_tools_file_search_type_only',
    'assistant_tools_function',
    'assistants_api_response_format_option',
    'assistants_api_tool_choice_option',
    'assistants_named_tool_choice',
    'attachment',
    'attachment_type',
    'audio_response_format',
    'audio_transcription',
    'audit_log',
    'audit_log_actor',
    'audit_log_actor_api_key',
    'audit_log_actor_service_account',
    'audit_log_actor_session',
    'audit_log_actor_user',
    'audit_log_event_type',
    'auto_chunking_strategy_request_param',
    'automatic_thread_titling_param',
    'batch',
    'batch_error',
    'batch_file_expiration_after',
    'batch_request_counts',
    'batch_request_input',
    'batch_request_output',
    'certificate',
    'chat_completion_allowed_tools',
    'chat_completion_allowed_tools_choice',
    'chat_completion_deleted',
    'chat_completion_function_call_option',
    'chat_completion_functions',
    'chat_completion_list',
    'chat_completion_message_custom_tool_call',
    'chat_completion_message_list',
    'chat_completion_message_tool_call',
    'chat_completion_message_tool_call_chunk',
    'chat_completion_message_tool_calls',
    'chat_completion_modalities',
    'chat_completion_named_tool_choice',
    'chat_completion_named_tool_choice_custom',
    'chat_completion_request_assistant_message',
    'chat_completion_request_assistant_message_content_part',
    'chat_completion_request_developer_message',
    'chat_completion_request_function_message',
    'chat_completion_request_message',
    'chat_completion_request_message_content_part_audio',
    'chat_completion_request_message_content_part_file',
    'chat_completion_request_message_content_part_image',
    'chat_completion_request_message_content_part_refusal',
    'chat_completion_request_message_content_part_text',
    'chat_completion_request_system_message',
    'chat_completion_request_system_message_content_part',
    'chat_completion_request_tool_message',
    'chat_completion_request_tool_message_content_part',
    'chat_completion_request_user_message',
    'chat_completion_request_user_message_content_part',
    'chat_completion_response_message',
    'chat_completion_role',
    'chat_completion_stream_options',
    'chat_completion_stream_response_delta',
    'chat_completion_token_logprob',
    'chat_completion_tool',
    'chat_completion_tool_choice_option',
    'chat_model',
    'chat_session_automatic_thread_titling',
    'chat_session_chatkit_configuration',
    'chat_session_file_upload',
    'chat_session_history',
    'chat_session_rate_limits',
    'chat_session_resource',
    'chat_session_status',
    'chatkit_configuration_param',
    'chatkit_workflow',
    'chatkit_workflow_tracing',
    'chunking_strategy_request_param',
    'chunking_strategy_response',
    'click_button_type',
    'click_param',
    'client_tool_call_item',
    'client_tool_call_status',
    'closed_status',
    'code_interpreter_container_auto',
    'code_interpreter_file_output',
    'code_interpreter_output_image',
    'code_interpreter_output_logs',
    'code_interpreter_text_output',
    'code_interpreter_tool',
    'code_interpreter_tool_call',
    'compact_resource',
    'compact_response_method_public_body',
    'compaction_body',
    'compaction_summary_item_param',
    'comparison_filter',
    'comparison_filter_value_items',
    'complete_upload_request',
    'completion_usage',
    'compound_filter',
    'computer_action',
    'computer_call_output_item_param',
    'computer_call_safety_check_param',
    'computer_environment',
    'computer_screenshot_content',
    'computer_screenshot_image',
    'computer_tool_call',
    'computer_tool_call_output',
    'computer_tool_call_output_resource',
    'computer_use_preview_tool',
    'container_file_citation_body',
    'container_file_list_resource',
    'container_file_resource',
    'container_list_resource',
    'container_memory_limit',
    'container_resource',
    'content',
    'conversation',
    'conversation_2',
    'conversation_item',
    'conversation_item_list',
    'conversation_param',
    'conversation_param_2',
    'conversation_resource',
    'costs_result',
    'create_assistant_request',
    'create_chat_completion_request',
    'create_chat_completion_response',
    'create_chat_completion_stream_response',
    'create_chat_session_body',
    'create_completion_request',
    'create_completion_response',
    'create_container_body',
    'create_container_file_body',
    'create_conversation_body',
    'create_embedding_request',
    'create_embedding_response',
    'create_eval_completions_run_data_source',
    'create_eval_custom_data_source_config',
    'create_eval_item',
    'create_eval_jsonl_run_data_source',
    'create_eval_label_model_grader',
    'create_eval_logs_data_source_config',
    'create_eval_request',
    'create_eval_responses_run_data_source',
    'create_eval_run_request',
    'create_eval_stored_completions_data_source_config',
    'create_file_request',
    'create_fine_tuning_checkpoint_permission_request',
    'create_fine_tuning_job_request',
    'create_group_body',
    'create_group_user_body',
    'create_image_edit_request',
    'create_image_request',
    'create_image_variation_request',
    'create_message_request',
    'create_model_response_properties',
    'create_moderation_request',
    'create_moderation_response',
    'create_response',
    'create_run_request',
    'create_run_request_without_stream',
    'create_speech_request',
    'create_speech_response_stream_event',
    'create_thread_and_run_request',
    'create_thread_and_run_request_without_stream',
    'create_thread_request',
    'create_transcription_request',
    'create_transcription_response_diarized_json',
    'create_transcription_response_json',
    'create_transcription_response_stream_event',
    'create_transcription_response_verbose_json',
    'create_translation_request',
    'create_translation_response_json',
    'create_translation_response_verbose_json',
    'create_upload_request',
    'create_vector_store_file_batch_request',
    'create_vector_store_file_request',
    'create_vector_store_request',
    'create_video_body',
    'create_video_remix_body',
    'create_voice_consent_request',
    'create_voice_request',
    'custom_grammar_format_param',
    'custom_text_format_param',
    'custom_tool_call',
    'custom_tool_call_output',
    'custom_tool_chat_completions',
    'custom_tool_param',
    'delete_assistant_response',
    'delete_certificate_response',
    'delete_file_response',
    'delete_fine_tuning_checkpoint_permission_response',
    'delete_message_response',
    'delete_model_response',
    'delete_thread_response',
    'delete_vector_store_file_response',
    'delete_vector_store_response',
    'deleted_conversation',
    'deleted_conversation_resource',
    'deleted_role_assignment_resource',
    'deleted_thread_resource',
    'deleted_video_resource',
    'detail_enum',
    'done_event',
    'double_click_action',
    'drag',
    'drag_point',
    'easy_input_message',
    'embedding',
    'error',
    'error_2',
    'error_event',
    'error_response',
    'eval',
    'eval_api_error',
    'eval_custom_data_source_config',
    'eval_grader_label_model',
    'eval_grader_python',
    'eval_grader_score_model',
    'eval_grader_string_check',
    'eval_grader_text_similarity',
    'eval_item',
    'eval_item_content',
    'eval_item_content_array',
    'eval_item_content_item',
    'eval_item_content_output_text',
    'eval_item_content_text',
    'eval_item_input_image',
    'eval_jsonl_file_content_source',
    'eval_jsonl_file_id_source',
    'eval_list',
    'eval_logs_data_source_config',
    'eval_responses_source',
    'eval_run',
    'eval_run_list',
    'eval_run_output_item',
    'eval_run_output_item_list',
    'eval_run_output_item_result',
    'eval_stored_completions_data_source_config',
    'eval_stored_completions_source',
    'expires_after_param',
    'file_annotation',
    'file_annotation_source',
    'file_citation_body',
    'file_expiration_after',
    'file_path',
    'file_purpose',
    'file_search_ranker',
    'file_search_ranking_options',
    'file_search_tool',
    'file_search_tool_call',
    'file_upload_param',
    'filters',
    'fine_tune_chat_completion_request_assistant_message',
    'fine_tune_chat_request_input',
    'fine_tune_dpo_hyperparameters',
    'fine_tune_dpo_method',
    'fine_tune_method',
    'fine_tune_preference_request_input',
    'fine_tune_reinforcement_hyperparameters',
    'fine_tune_reinforcement_method',
    'fine_tune_reinforcement_request_input',
    'fine_tune_supervised_hyperparameters',
    'fine_tune_supervised_method',
    'fine_tuning_checkpoint_permission',
    'fine_tuning_integration',
    'fine_tuning_job',
    'fine_tuning_job_checkpoint',
    'fine_tuning_job_event',
    'function_and_custom_tool_call_output',
    'function_call_item_status',
    'function_call_output_item_param',
    'function_object',
    'function_parameters',
    'function_shell_action',
    'function_shell_action_param',
    'function_shell_call',
    'function_shell_call_item_param',
    'function_shell_call_item_status',
    'function_shell_call_output',
    'function_shell_call_output_content',
    'function_shell_call_output_content_param',
    'function_shell_call_output_exit_outcome',
    'function_shell_call_output_exit_outcome_param',
    'function_shell_call_output_item_param',
    'function_shell_call_output_outcome_param',
    'function_shell_call_output_timeout_outcome',
    'function_shell_call_output_timeout_outcome_param',
    'function_shell_tool_param',
    'function_tool',
    'function_tool_call',
    'function_tool_call_output',
    'function_tool_call_output_resource',
    'function_tool_call_resource',
    'grader_label_model',
    'grader_multi',
    'grader_python',
    'grader_score_model',
    'grader_string_check',
    'grader_text_similarity',
    'grammar_syntax1',
    'group',
    'group_deleted_resource',
    'group_list_resource',
    'group_resource_with_success',
    'group_response',
    'group_role_assignment',
    'group_user_assignment',
    'group_user_deleted_resource',
    'history_param',
    'hybrid_search_options',
    'image',
    'image_detail',
    'image_edit_completed_event',
    'image_edit_partial_image_event',
    'image_edit_stream_event',
    'image_gen_completed_event',
    'image_gen_input_usage_details',
    'image_gen_output_tokens_details',
    'image_gen_partial_image_event',
    'image_gen_stream_event',
    'image_gen_tool',
    'image_gen_tool_call',
    'image_gen_usage',
    'images_response',
    'images_usage',
    'include_enum',
    'inference_options',
    'input_audio',
    'input_content',
    'input_fidelity',
    'input_file_content',
    'input_file_content_param',
    'input_image_content',
    'input_image_content_param_auto_param',
    'input_item',
    'input_message',
    'input_message_content_list',
    'input_message_resource',
    'input_param',
    'input_text_content',
    'input_text_content_param',
    'invite',
    'invite_delete_response',
    'invite_list_response',
    'invite_project_group_body',
    'invite_request',
    'item',
    'item_field',
    'item_reference_param',
    'item_resource',
    'key_press_action',
    'list_assistants_response',
    'list_audit_logs_response',
    'list_batches_response',
    'list_certificates_response',
    'list_files_response',
    'list_fine_tuning_checkpoint_permission_response',
    'list_fine_tuning_job_checkpoints_response',
    'list_fine_tuning_job_events_response',
    'list_messages_response',
    'list_models_response',
    'list_paginated_fine_tuning_jobs_response',
    'list_run_steps_response',
    'list_runs_response',
    'list_vector_store_files_response',
    'list_vector_stores_response',
    'local_shell_call_status',
    'local_shell_exec_action',
    'local_shell_tool_call',
    'local_shell_tool_call_output',
    'local_shell_tool_param',
    'locked_status',
    'log_prob',
    'log_prob_properties',
    'mcp_approval_request',
    'mcp_approval_response',
    'mcp_approval_response_resource',
    'mcp_list_tools',
    'mcp_list_tools_tool',
    'mcp_tool',
    'mcp_tool_call',
    'mcp_tool_call_status',
    'mcp_tool_filter',
    'message',
    'message_content',
    'message_content_delta',
    'message_content_image_file_object',
    'message_content_image_url_object',
    'message_content_refusal_object',
    'message_content_text_annotations_file_citation_object',
    'message_content_text_annotations_file_path_object',
    'message_content_text_object',
    'message_delta_content_image_file_object',
    'message_delta_content_image_url_object',
    'message_delta_content_refusal_object',
    'message_delta_content_text_annotations_file_citation_object',
    'message_delta_content_text_annotations_file_path_object',
    'message_delta_content_text_object',
    'message_delta_object',
    'message_object',
    'message_request_content_text_object',
    'message_role',
    'message_status',
    'message_stream_event',
    'metadata',
    'model',
    'model_ids',
    'model_ids_compaction',
    'model_ids_responses',
    'model_ids_shared',
    'model_response_properties',
    'moderation_image_url_input',
    'moderation_text_input',
    'modify_assistant_request',
    'modify_certificate_request',
    'modify_message_request',
    'modify_run_request',
    'modify_thread_request',
    'move',
    'noise_reduction_type',
    'open_ai_file',
    'order_enum',
    'other_chunking_strategy_response_param',
    'output_audio',
    'output_content',
    'output_item',
    'output_message',
    'output_message_content',
    'output_text_content',
    'parallel_tool_calls',
    'partial_images',
    'prediction_content',
    'project',
    'project_api_key',
    'project_api_key_delete_response',
    'project_api_key_list_response',
    'project_create_request',
    'project_group',
    'project_group_deleted_resource',
    'project_group_list_resource',
    'project_list_response',
    'project_rate_limit',
    'project_rate_limit_list_response',
    'project_rate_limit_update_request',
    'project_service_account',
    'project_service_account_api_key',
    'project_service_account_create_request',
    'project_service_account_create_response',
    'project_service_account_delete_response',
    'project_service_account_list_response',
    'project_update_request',
    'project_user',
    'project_user_create_request',
    'project_user_delete_response',
    'project_user_list_response',
    'project_user_update_request',
    'prompt',
    'public_assign_organization_group_role_body',
    'public_create_organization_role_body',
    'public_role_list_resource',
    'public_update_organization_role_body',
    'ranker_version_type',
    'ranking_options',
    'rate_limits_param',
    'realtime_audio_formats',
    'realtime_beta_client_event_conversation_item_create',
    'realtime_beta_client_event_conversation_item_delete',
    'realtime_beta_client_event_conversation_item_retrieve',
    'realtime_beta_client_event_conversation_item_truncate',
    'realtime_beta_client_event_input_audio_buffer_append',
    'realtime_beta_client_event_input_audio_buffer_clear',
    'realtime_beta_client_event_input_audio_buffer_commit',
    'realtime_beta_client_event_output_audio_buffer_clear',
    'realtime_beta_client_event_response_cancel',
    'realtime_beta_client_event_response_create',
    'realtime_beta_client_event_session_update',
    'realtime_beta_client_event_transcription_session_update',
    'realtime_beta_response',
    'realtime_beta_response_create_params',
    'realtime_beta_server_event_conversation_item_created',
    'realtime_beta_server_event_conversation_item_deleted',
    'realtime_beta_server_event_conversation_item_input_audio_transcription_completed',
    'realtime_beta_server_event_conversation_item_input_audio_transcription_delta',
    'realtime_beta_server_event_conversation_item_input_audio_transcription_failed',
    'realtime_beta_server_event_conversation_item_input_audio_transcription_segment',
    'realtime_beta_server_event_conversation_item_retrieved',
    'realtime_beta_server_event_conversation_item_truncated',
    'realtime_beta_server_event_error',
    'realtime_beta_server_event_input_audio_buffer_cleared',
    'realtime_beta_server_event_input_audio_buffer_committed',
    'realtime_beta_server_event_input_audio_buffer_speech_started',
    'realtime_beta_server_event_input_audio_buffer_speech_stopped',
    'realtime_beta_server_event_mcp_list_tools_completed',
    'realtime_beta_server_event_mcp_list_tools_failed',
    'realtime_beta_server_event_mcp_list_tools_in_progress',
    'realtime_beta_server_event_rate_limits_updated',
    'realtime_beta_server_event_response_audio_delta',
    'realtime_beta_server_event_response_audio_done',
    'realtime_beta_server_event_response_audio_transcript_delta',
    'realtime_beta_server_event_response_audio_transcript_done',
    'realtime_beta_server_event_response_content_part_added',
    'realtime_beta_server_event_response_content_part_done',
    'realtime_beta_server_event_response_created',
    'realtime_beta_server_event_response_done',
    'realtime_beta_server_event_response_function_call_arguments_delta',
    'realtime_beta_server_event_response_function_call_arguments_done',
    'realtime_beta_server_event_response_mcp_call_arguments_delta',
    'realtime_beta_server_event_response_mcp_call_arguments_done',
    'realtime_beta_server_event_response_mcp_call_completed',
    'realtime_beta_server_event_response_mcp_call_failed',
    'realtime_beta_server_event_response_mcp_call_in_progress',
    'realtime_beta_server_event_response_output_item_added',
    'realtime_beta_server_event_response_output_item_done',
    'realtime_beta_server_event_response_text_delta',
    'realtime_beta_server_event_response_text_done',
    'realtime_beta_server_event_session_created',
    'realtime_beta_server_event_session_updated',
    'realtime_beta_server_event_transcription_session_created',
    'realtime_beta_server_event_transcription_session_updated',
    'realtime_call_create_request',
    'realtime_call_refer_request',
    'realtime_call_reject_request',
    'realtime_client_event',
    'realtime_client_event_conversation_item_create',
    'realtime_client_event_conversation_item_delete',
    'realtime_client_event_conversation_item_retrieve',
    'realtime_client_event_conversation_item_truncate',
    'realtime_client_event_input_audio_buffer_append',
    'realtime_client_event_input_audio_buffer_clear',
    'realtime_client_event_input_audio_buffer_commit',
    'realtime_client_event_output_audio_buffer_clear',
    'realtime_client_event_response_cancel',
    'realtime_client_event_response_create',
    'realtime_client_event_session_update',
    'realtime_client_event_transcription_session_update',
    'realtime_connect_params',
    'realtime_conversation_item',
    'realtime_conversation_item_function_call',
    'realtime_conversation_item_function_call_output',
    'realtime_conversation_item_message_assistant',
    'realtime_conversation_item_message_system',
    'realtime_conversation_item_message_user',
    'realtime_conversation_item_with_reference',
    'realtime_create_client_secret_request',
    'realtime_create_client_secret_response',
    'realtime_function_tool',
    'realtime_mcp_approval_request',
    'realtime_mcp_approval_response',
    'realtime_mcp_list_tools',
    'realtime_mcp_protocol_error',
    'realtime_mcp_tool_call',
    'realtime_mcp_tool_execution_error',
    'realtime_mcphttp_error',
    'realtime_response',
    'realtime_response_create_params',
    'realtime_server_event',
    'realtime_server_event_conversation_created',
    'realtime_server_event_conversation_item_added',
    'realtime_server_event_conversation_item_created',
    'realtime_server_event_conversation_item_deleted',
    'realtime_server_event_conversation_item_done',
    'realtime_server_event_conversation_item_input_audio_transcription_completed',
    'realtime_server_event_conversation_item_input_audio_transcription_delta',
    'realtime_server_event_conversation_item_input_audio_transcription_failed',
    'realtime_server_event_conversation_item_input_audio_transcription_segment',
    'realtime_server_event_conversation_item_retrieved',
    'realtime_server_event_conversation_item_truncated',
    'realtime_server_event_error',
    'realtime_server_event_input_audio_buffer_cleared',
    'realtime_server_event_input_audio_buffer_committed',
    'realtime_server_event_input_audio_buffer_dtmf_event_received',
    'realtime_server_event_input_audio_buffer_speech_started',
    'realtime_server_event_input_audio_buffer_speech_stopped',
    'realtime_server_event_input_audio_buffer_timeout_triggered',
    'realtime_server_event_mcp_list_tools_completed',
    'realtime_server_event_mcp_list_tools_failed',
    'realtime_server_event_mcp_list_tools_in_progress',
    'realtime_server_event_output_audio_buffer_cleared',
    'realtime_server_event_output_audio_buffer_started',
    'realtime_server_event_output_audio_buffer_stopped',
    'realtime_server_event_rate_limits_updated',
    'realtime_server_event_response_audio_delta',
    'realtime_server_event_response_audio_done',
    'realtime_server_event_response_audio_transcript_delta',
    'realtime_server_event_response_audio_transcript_done',
    'realtime_server_event_response_content_part_added',
    'realtime_server_event_response_content_part_done',
    'realtime_server_event_response_created',
    'realtime_server_event_response_done',
    'realtime_server_event_response_function_call_arguments_delta',
    'realtime_server_event_response_function_call_arguments_done',
    'realtime_server_event_response_mcp_call_arguments_delta',
    'realtime_server_event_response_mcp_call_arguments_done',
    'realtime_server_event_response_mcp_call_completed',
    'realtime_server_event_response_mcp_call_failed',
    'realtime_server_event_response_mcp_call_in_progress',
    'realtime_server_event_response_output_item_added',
    'realtime_server_event_response_output_item_done',
    'realtime_server_event_response_text_delta',
    'realtime_server_event_response_text_done',
    'realtime_server_event_session_created',
    'realtime_server_event_session_updated',
    'realtime_server_event_transcription_session_updated',
    'realtime_session',
    'realtime_session_create_request',
    'realtime_session_create_request_ga',
    'realtime_session_create_response',
    'realtime_session_create_response_ga',
    'realtime_transcription_session_create_request',
    'realtime_transcription_session_create_request_ga',
    'realtime_transcription_session_create_response',
    'realtime_transcription_session_create_response_ga',
    'realtime_truncation',
    'realtime_turn_detection',
    'reasoning',
    'reasoning_effort',
    'reasoning_item',
    'reasoning_text_content',
    'refusal_content',
    'response',
    'response_audio_delta_event',
    'response_audio_done_event',
    'response_audio_transcript_delta_event',
    'response_audio_transcript_done_event',
    'response_code_interpreter_call_code_delta_event',
    'response_code_interpreter_call_code_done_event',
    'response_code_interpreter_call_completed_event',
    'response_code_interpreter_call_in_progress_event',
    'response_code_interpreter_call_interpreting_event',
    'response_completed_event',
    'response_content_part_added_event',
    'response_content_part_done_event',
    'response_created_event',
    'response_custom_tool_call_input_delta_event',
    'response_custom_tool_call_input_done_event',
    'response_error',
    'response_error_code',
    'response_error_event',
    'response_failed_event',
    'response_file_search_call_completed_event',
    'response_file_search_call_in_progress_event',
    'response_file_search_call_searching_event',
    'response_format_json_object',
    'response_format_json_schema',
    'response_format_json_schema_schema',
    'response_format_text',
    'response_format_text_grammar',
    'response_format_text_python',
    'response_function_call_arguments_delta_event',
    'response_function_call_arguments_done_event',
    'response_image_gen_call_completed_event',
    'response_image_gen_call_generating_event',
    'response_image_gen_call_in_progress_event',
    'response_image_gen_call_partial_image_event',
    'response_in_progress_event',
    'response_incomplete_event',
    'response_item_list',
    'response_log_prob',
    'response_mcp_call_arguments_delta_event',
    'response_mcp_call_arguments_done_event',
    'response_mcp_call_completed_event',
    'response_mcp_call_failed_event',
    'response_mcp_call_in_progress_event',
    'response_mcp_list_tools_completed_event',
    'response_mcp_list_tools_failed_event',
    'response_mcp_list_tools_in_progress_event',
    'response_modalities',
    'response_output_item_added_event',
    'response_output_item_done_event',
    'response_output_text',
    'response_output_text_annotation_added_event',
    'response_prompt_variables',
    'response_properties',
    'response_queued_event',
    'response_reasoning_summary_part_added_event',
    'response_reasoning_summary_part_done_event',
    'response_reasoning_summary_text_delta_event',
    'response_reasoning_summary_text_done_event',
    'response_reasoning_text_delta_event',
    'response_reasoning_text_done_event',
    'response_refusal_delta_event',
    'response_refusal_done_event',
    'response_stream_event',
    'response_stream_options',
    'response_text_delta_event',
    'response_text_done_event',
    'response_text_param',
    'response_usage',
    'response_web_search_call_completed_event',
    'response_web_search_call_in_progress_event',
    'response_web_search_call_searching_event',
    'role',
    'role_deleted_resource',
    'role_list_resource',
    'run_completion_usage',
    'run_grader_request',
    'run_grader_response',
    'run_object',
    'run_status',
    'run_step_completion_usage',
    'run_step_delta_object',
    'run_step_delta_object_delta',
    'run_step_delta_step_details_message_creation_object',
    'run_step_delta_step_details_tool_call',
    'run_step_delta_step_details_tool_calls_code_object',
    'run_step_delta_step_details_tool_calls_code_output_image_object',
    'run_step_delta_step_details_tool_calls_code_output_logs_object',
    'run_step_delta_step_details_tool_calls_file_search_object',
    'run_step_delta_step_details_tool_calls_function_object',
    'run_step_delta_step_details_tool_calls_object',
    'run_step_details_message_creation_object',
    'run_step_details_tool_call',
    'run_step_details_tool_calls_code_object',
    'run_step_details_tool_calls_code_output_image_object',
    'run_step_details_tool_calls_code_output_logs_object',
    'run_step_details_tool_calls_file_search_object',
    'run_step_details_tool_calls_file_search_ranking_options_object',
    'run_step_details_tool_calls_file_search_result_object',
    'run_step_details_tool_calls_function_object',
    'run_step_details_tool_calls_object',
    'run_step_object',
    'run_step_stream_event',
    'run_stream_event',
    'run_tool_call_object',
    'screenshot',
    'scroll',
    'search_context_size',
    'service_tier',
    'specific_apply_patch_param',
    'specific_function_shell_param',
    'speech_audio_delta_event',
    'speech_audio_done_event',
    'static_chunking_strategy',
    'static_chunking_strategy_request_param',
    'static_chunking_strategy_response_param',
    'stop_configuration',
    'submit_tool_outputs_run_request',
    'submit_tool_outputs_run_request_without_stream',
    'summary',
    'summary_text_content',
    'task_group_item',
    'task_group_task',
    'task_item',
    'task_type',
    'text_annotation',
    'text_annotation_delta',
    'text_content',
    'text_response_format_configuration',
    'text_response_format_json_schema',
    'thread_item',
    'thread_item_list_resource',
    'thread_list_resource',
    'thread_object',
    'thread_resource',
    'thread_stream_event',
    'toggle_certificates_request',
    'token_counts_body',
    'token_counts_resource',
    'tool',
    'tool_choice',
    'tool_choice_allowed',
    'tool_choice_custom',
    'tool_choice_function',
    'tool_choice_mcp',
    'tool_choice_options',
    'tool_choice_param',
    'tool_choice_types',
    'tools_array',
    'top_log_prob',
    'transcript_text_delta_event',
    'transcript_text_done_event',
    'transcript_text_segment_event',
    'transcript_text_usage_duration',
    'transcript_text_usage_tokens',
    'transcription_chunking_strategy',
    'transcription_diarized_segment',
    'transcription_include',
    'transcription_segment',
    'transcription_word',
    'truncation_enum',
    'truncation_object',
    'type',
    'update_conversation_body',
    'update_group_body',
    'update_vector_store_file_attributes_request',
    'update_vector_store_request',
    'update_voice_consent_request',
    'upload',
    'upload_certificate_request',
    'upload_part',
    'url_annotation',
    'url_annotation_source',
    'url_citation_body',
    'usage_audio_speeches_result',
    'usage_audio_transcriptions_result',
    'usage_code_interpreter_sessions_result',
    'usage_completions_result',
    'usage_embeddings_result',
    'usage_images_result',
    'usage_moderations_result',
    'usage_response',
    'usage_time_bucket',
    'usage_vector_stores_result',
    'user',
    'user_delete_response',
    'user_list_resource',
    'user_list_response',
    'user_message_input_text',
    'user_message_item',
    'user_message_quoted_text',
    'user_role_assignment',
    'user_role_update_request',
    'vad_config',
    'validate_grader_request',
    'validate_grader_response',
    'vector_store_expiration_after',
    'vector_store_file_attributes',
    'vector_store_file_batch_object',
    'vector_store_file_content_response',
    'vector_store_file_object',
    'vector_store_object',
    'vector_store_search_request',
    'vector_store_search_result_content_object',
    'vector_store_search_result_item',
    'vector_store_search_results_page',
    'verbosity',
    'video_content_variant',
    'video_list_resource',
    'video_model',
    'video_resource',
    'video_seconds',
    'video_size',
    'video_status',
    'voice_consent_deleted_resource',
    'voice_consent_list_resource',
    'voice_consent_resource',
    'voice_ids_or_custom_voice',
    'voice_ids_shared',
    'voice_resource',
    'wait',
    'web_search_action_find',
    'web_search_action_open_page',
    'web_search_action_search',
    'web_search_approximate_location',
    'web_search_context_size',
    'web_search_location',
    'web_search_preview_tool',
    'web_search_tool',
    'web_search_tool_call',
    'webhook_batch_cancelled',
    'webhook_batch_completed',
    'webhook_batch_expired',
    'webhook_batch_failed',
    'webhook_eval_run_canceled',
    'webhook_eval_run_failed',
    'webhook_eval_run_succeeded',
    'webhook_fine_tuning_job_cancelled',
    'webhook_fine_tuning_job_failed',
    'webhook_fine_tuning_job_succeeded',
    'webhook_realtime_call_incoming',
    'webhook_response_cancelled',
    'webhook_response_completed',
    'webhook_response_failed',
    'webhook_response_incomplete',
    'widget_message_item',
    'workflow_param',
    'workflow_tracing_param',
    'ActiveStatus',
    'AddUploadPartRequest',
    'AdminApiKey',
    'Owner',
    'ApiKeyList',
    'ApplyPatchCallOutputStatus',
    'ApplyPatchCallOutputStatusParam',
    'ApplyPatchCallStatus',
    'ApplyPatchCallStatusParam',
    'ApplyPatchCreateFileOperation',
    'ApplyPatchCreateFileOperationParam',
    'ApplyPatchDeleteFileOperation',
    'ApplyPatchDeleteFileOperationParam',
    'ApplyPatchToolCall',
    'ApplyPatchToolCallItemParam',
    'ApplyPatchToolCallOutput',
    'ApplyPatchToolCallOutputItemParam',
    'ApplyPatchToolParam',
    'ApplyPatchUpdateFileOperation',
    'ApplyPatchUpdateFileOperationParam',
    'ApproximateLocation',
    'AssignedRoleDetails',
    'AssistantMessageItem',
    'AssistantObject',
    'CodeInterpreter',
    'FileSearch',
    'ToolResources',
    'AssistantSupportedModels',
    'AssistantToolsCode',
    'AssistantToolsFileSearch',
    'FileSearch',
    'AssistantToolsFileSearchTypeOnly',
    'AssistantToolsFunction',
    'AssistantsNamedToolChoice',
    'Function',
    'Attachment',
    'AttachmentType',
    'AudioResponseFormat',
    'AudioTranscription',
    'ApiKeyCreated',
    'ApiKeyDeleted',
    'ApiKeyUpdated',
    'AuditLog',
    'CertificateCreated',
    'CertificateDeleted',
    'CertificateUpdated',
    'CertificatesActivated',
    'CertificatesDeactivated',
    'CertificatesItem',
    'ChangesRequested',
    'CheckpointPermissionCreated',
    'CheckpointPermissionDeleted',
    'ConfigsItem',
    'Data',
    'ExternalKeyRegistered',
    'ExternalKeyRemoved',
    'GroupCreated',
    'GroupDeleted',
    'GroupUpdated',
    'InviteAccepted',
    'InviteDeleted',
    'InviteSent',
    'IpAllowlistConfigActivated',
    'IpAllowlistConfigDeactivated',
    'IpAllowlistCreated',
    'IpAllowlistDeleted',
    'IpAllowlistUpdated',
    'LoginFailed',
    'LogoutFailed',
    'OrganizationUpdated',
    'Project',
    'ProjectArchived',
    'ProjectCreated',
    'ProjectDeleted',
    'ProjectUpdated',
    'RateLimitDeleted',
    'RateLimitUpdated',
    'RoleAssignmentCreated',
    'RoleAssignmentDeleted',
    'RoleCreated',
    'RoleDeleted',
    'RoleUpdated',
    'ScimDisabled',
    'ScimEnabled',
    'ServiceAccountCreated',
    'ServiceAccountDeleted',
    'ServiceAccountUpdated',
    'UserAdded',
    'UserDeleted',
    'UserUpdated',
    'AuditLogActor',
    'AuditLogActorApiKey',
    'AuditLogActorServiceAccount',
    'AuditLogActorSession',
    'AuditLogActorUser',
    'AuditLogEventType',
    'AutoChunkingStrategyRequestParam',
    'AutomaticThreadTitlingParam',
    'Batch',
    'Errors',
    'InputTokensDetails',
    'OutputTokensDetails',
    'Usage',
    'BatchError',
    'BatchFileExpirationAfter',
    'BatchRequestCounts',
    'BatchRequestInput',
    'BatchRequestOutput',
    'Error',
    'Response',
    'Certificate',
    'CertificateDetails',
    'ChatCompletionAllowedTools',
    'ChatCompletionAllowedToolsChoice',
    'ChatCompletionDeleted',
    'ChatCompletionFunctionCallOption',
    'ChatCompletionFunctions',
    'ChatCompletionList',
    'ChatCompletionMessageCustomToolCall',
    'Custom',
    'ChatCompletionMessageList',
    'DataItem',
    'ChatCompletionMessageToolCall',
    'Function',
    'ChatCompletionMessageToolCallChunk',
    'Function',
    'ChatCompletionMessageToolCalls',
    'ChatCompletionModalities',
    'ChatCompletionNamedToolChoice',
    'Function',
    'ChatCompletionNamedToolChoiceCustom',
    'Custom',
    'Audio',
    'ChatCompletionRequestAssistantMessage',
    'FunctionCall',
    'ChatCompletionRequestDeveloperMessage',
    'ChatCompletionRequestFunctionMessage',
    'Audio',
    'FunctionCall',
    'ChatCompletionRequestMessageContentPartAudio',
    'InputAudio',
    'ChatCompletionRequestMessageContentPartFile',
    'File',
    'ChatCompletionRequestMessageContentPartImage',
    'ImageUrl',
    'ChatCompletionRequestMessageContentPartRefusal',
    'ChatCompletionRequestMessageContentPartText',
    'ChatCompletionRequestSystemMessage',
    'ChatCompletionRequestToolMessage',
    'ChatCompletionRequestUserMessage',
    'AnnotationsItem',
    'Audio',
    'ChatCompletionResponseMessage',
    'FunctionCall',
    'UrlCitation',
    'ChatCompletionRole',
    'ChatCompletionStreamOptions',
    'ChatCompletionStreamResponseDelta',
    'FunctionCall',
    'ChatCompletionTokenLogprob',
    'TopLogprobsItem',
    'ChatCompletionTool',
    'ChatModel',
    'ChatSessionAutomaticThreadTitling',
    'ChatSessionChatkitConfiguration',
    'ChatSessionFileUpload',
    'ChatSessionHistory',
    'ChatSessionRateLimits',
    'ChatSessionResource',
    'ChatSessionStatus',
    'ChatkitConfigurationParam',
    'ChatkitWorkflow',
    'ChatkitWorkflowTracing',
    'ClickButtonType',
    'ClickParam',
    'ClientToolCallItem',
    'ClientToolCallStatus',
    'ClosedStatus',
    'CodeInterpreterContainerAuto',
    'CodeInterpreterFileOutput',
    'FilesItem',
    'CodeInterpreterOutputImage',
    'CodeInterpreterOutputLogs',
    'CodeInterpreterTextOutput',
    'CodeInterpreterTool',
    'CodeInterpreterToolCall',
    'CompactResource',
    'InputTokensDetails',
    'OutputTokensDetails',
    'CompactResponseMethodPublicBody',
    'CompactionBody',
    'CompactionSummaryItemParam',
    'ComparisonFilter',
    'ComparisonFilterValueItems',
    'CompleteUploadRequest',
    'CompletionTokensDetails',
    'CompletionUsage',
    'PromptTokensDetails',
    'CompoundFilter',
    'ComputerCallOutputItemParam',
    'ComputerCallSafetyCheckParam',
    'ComputerEnvironment',
    'ComputerScreenshotContent',
    'ComputerScreenshotImage',
    'ComputerToolCall',
    'ComputerToolCallOutput',
    'ComputerToolCallOutputResource',
    'ComputerUsePreviewTool',
    'ContainerFileCitationBody',
    'ContainerFileListResource',
    'ContainerFileResource',
    'ContainerListResource',
    'ContainerMemoryLimit',
    'ContainerResource',
    'ExpiresAfter',
    'Conversation',
    'Conversation2',
    'ResultsOption0Item',
    'ConversationItemList',
    'ConversationParam2',
    'ConversationResource',
    'Amount',
    'CostsResult',
    'AutoChunkingStrategy',
    'CodeInterpreter',
    'CreateAssistantRequest',
    'FileSearch',
    'Static',
    'StaticChunkingStrategy',
    'ToolResources',
    'VectorStoresItem',
    'Audio',
    'CreateChatCompletionRequest',
    'UserLocation',
    'WebSearchOptions',
    'ChoicesItem',
    'CompletionTokensDetails',
    'CreateChatCompletionResponse',
    'Logprobs',
    'PromptTokensDetails',
    'ChoicesItem',
    'CompletionTokensDetails',
    'CreateChatCompletionStreamResponse',
    'Logprobs',
    'PromptTokensDetails',
    'CreateChatSessionBody',
    'CreateCompletionRequest',
    'ChoicesItem',
    'CompletionTokensDetails',
    'CreateCompletionResponse',
    'Logprobs',
    'PromptTokensDetails',
    'CreateContainerBody',
    'ExpiresAfter',
    'CreateContainerFileBody',
    'CreateConversationBody',
    'CreateEmbeddingRequest',
    'CreateEmbeddingResponse',
    'Usage',
    'CreateEvalCompletionsRunDataSource',
    'ItemReferenceInputMessages',
    'SamplingParams',
    'TemplateInputMessages',
    'CreateEvalCustomDataSourceConfig',
    'CreateEvalJsonlRunDataSource',
    'CreateEvalLabelModelGrader',
    'CreateEvalLogsDataSourceConfig',
    'CreateEvalRequest',
    'ChatMessage',
    'CreateEvalResponsesRunDataSource',
    'InputMessagesItemReference',
    'InputMessagesTemplate',
    'SamplingParams',
    'Text',
    'CreateEvalRunRequest',
    'CreateEvalStoredCompletionsDataSourceConfig',
    'CreateFileRequest',
    'CreateFineTuningCheckpointPermissionRequest',
    'CreateFineTuningJobRequest',
    'Hyperparameters',
    'IntegrationsItem',
    'Wandb',
    'CreateGroupBody',
    'CreateGroupUserBody',
    'CreateImageEditRequest',
    'CreateImageRequest',
    'CreateImageVariationRequest',
    'AttachmentsOption0Item',
    'CreateMessageRequest',
    'CreateModelResponseProperties',
    'CreateModerationRequest',
    'Categories',
    'CategoryAppliedInputTypes',
    'CategoryScores',
    'CreateModerationResponse',
    'ResultsItem',
    'CreateResponse',
    'CreateRunRequest',
    'CreateRunRequestWithoutStream',
    'CreateSpeechRequest',
    'AutoChunkingStrategy',
    'CodeInterpreter',
    'CreateThreadAndRunRequest',
    'FileSearch',
    'Static',
    'StaticChunkingStrategy',
    'ToolResources',
    'VectorStoresItem',
    'AutoChunkingStrategy',
    'CodeInterpreter',
    'CreateThreadAndRunRequestWithoutStream',
    'FileSearch',
    'Static',
    'StaticChunkingStrategy',
    'ToolResources',
    'VectorStoresItem',
    'AutoChunkingStrategy',
    'CodeInterpreter',
    'CreateThreadRequest',
    'FileSearch',
    'Static',
    'StaticChunkingStrategy',
    'ToolResources',
    'VectorStoresItem',
    'CreateTranscriptionRequest',
    'CreateTranscriptionResponseDiarizedJson',
    'CreateTranscriptionResponseJson',
    'LogprobsItem',
    'InputTokenDetails',
    'CreateTranscriptionResponseVerboseJson',
    'CreateTranslationRequest',
    'CreateTranslationResponseJson',
    'CreateTranslationResponseVerboseJson',
    'CreateUploadRequest',
    'CreateVectorStoreFileBatchRequest',
    'CreateVectorStoreFileRequest',
    'CreateVectorStoreRequest',
    'CreateVideoBody',
    'CreateVideoRemixBody',
    'CreateVoiceConsentRequest',
    'CreateVoiceRequest',
    'CustomGrammarFormatParam',
    'CustomTextFormatParam',
    'CustomToolCall',
    'CustomToolCallOutput',
    'Custom',
    'CustomToolChatCompletions',
    'Grammar',
    'GrammarFormat',
    'TextFormat',
    'CustomToolParam',
    'DeleteAssistantResponse',
    'DeleteCertificateResponse',
    'DeleteFileResponse',
    'DeleteFineTuningCheckpointPermissionResponse',
    'DeleteMessageResponse',
    'DeleteModelResponse',
    'DeleteThreadResponse',
    'DeleteVectorStoreFileResponse',
    'DeleteVectorStoreResponse',
    'DeletedConversation',
    'DeletedConversationResource',
    'DeletedRoleAssignmentResource',
    'DeletedThreadResource',
    'DeletedVideoResource',
    'DetailEnum',
    'DoneEvent',
    'DoubleClickAction',
    'Drag',
    'DragPoint',
    'EasyInputMessage',
    'Embedding',
    'Error',
    'Error2',
    'ErrorEvent',
    'ErrorResponse',
    'Eval',
    'EvalApiError',
    'EvalCustomDataSourceConfig',
    'EvalGraderLabelModel',
    'EvalGraderPython',
    'EvalGraderScoreModel',
    'EvalGraderStringCheck',
    'EvalGraderTextSimilarity',
    'EvalItem',
    'EvalItemContentArray',
    'EvalItemContentOutputText',
    'EvalItemContentText',
    'EvalItemInputImage',
    'ContentItem',
    'EvalJsonlFileContentSource',
    'EvalJsonlFileIdSource',
    'EvalList',
    'EvalLogsDataSourceConfig',
    'EvalResponsesSource',
    'EvalRun',
    'PerModelUsageItem',
    'PerTestingCriteriaResultsItem',
    'ResultCounts',
    'EvalRunList',
    'EvalRunOutputItem',
    'InputItem',
    'OutputItem',
    'Sample',
    'Usage',
    'EvalRunOutputItemList',
    'EvalRunOutputItemResult',
    'EvalStoredCompletionsDataSourceConfig',
    'EvalStoredCompletionsSource',
    'ExpiresAfterParam',
    'FileAnnotation',
    'FileAnnotationSource',
    'FileCitationBody',
    'FileExpirationAfter',
    'FilePath',
    'FilePurpose',
    'FileSearchRanker',
    'FileSearchRankingOptions',
    'FileSearchTool',
    'FileSearchToolCall',
    'ResultsOption0Item',
    'FileUploadParam',
    'Audio',
    'FineTuneChatCompletionRequestAssistantMessage',
    'FunctionCall',
    'FineTuneChatRequestInput',
    'FineTuneDPOHyperparameters',
    'FineTuneDPOMethod',
    'FineTuneMethod',
    'FineTunePreferenceRequestInput',
    'Input',
    'FineTuneReinforcementHyperparameters',
    'FineTuneReinforcementMethod',
    'FineTuneReinforcementRequestInput',
    'FineTuneSupervisedHyperparameters',
    'FineTuneSupervisedMethod',
    'FineTuningCheckpointPermission',
    'FineTuningIntegration',
    'Wandb',
    'Error',
    'FineTuningJob',
    'Hyperparameters',
    'FineTuningJobCheckpoint',
    'Metrics',
    'FineTuningJobEvent',
    'FunctionCallItemStatus',
    'FunctionCallOutputItemParam',
    'FunctionObject',
    'FunctionParameters',
    'FunctionShellAction',
    'FunctionShellActionParam',
    'FunctionShellCall',
    'FunctionShellCallItemParam',
    'FunctionShellCallItemStatus',
    'FunctionShellCallOutput',
    'FunctionShellCallOutputContent',
    'FunctionShellCallOutputContentParam',
    'FunctionShellCallOutputExitOutcome',
    'FunctionShellCallOutputExitOutcomeParam',
    'FunctionShellCallOutputItemParam',
    'FunctionShellCallOutputTimeoutOutcome',
    'FunctionShellCallOutputTimeoutOutcomeParam',
    'FunctionShellToolParam',
    'FunctionTool',
    'FunctionToolCall',
    'FunctionToolCallOutput',
    'FunctionToolCallOutputResource',
    'FunctionToolCallResource',
    'GraderLabelModel',
    'GraderMulti',
    'GraderPython',
    'GraderScoreModel',
    'SamplingParams',
    'GraderStringCheck',
    'GraderTextSimilarity',
    'GrammarSyntax1',
    'Group',
    'GroupDeletedResource',
    'GroupListResource',
    'GroupResourceWithSuccess',
    'GroupResponse',
    'GroupRoleAssignment',
    'GroupUserAssignment',
    'GroupUserDeletedResource',
    'HistoryParam',
    'HybridSearchOptions',
    'Image',
    'ImageDetail',
    'ImageEditCompletedEvent',
    'InputTokensDetails',
    'ImageEditPartialImageEvent',
    'InputTokensDetails',
    'ImageGenCompletedEvent',
    'InputTokensDetails',
    'ImageGenInputUsageDetails',
    'ImageGenOutputTokensDetails',
    'ImageGenPartialImageEvent',
    'InputTokensDetails',
    'ImageGenTool',
    'InputImageMask',
    'ImageGenToolCall',
    'ImageGenUsage',
    'ImagesResponse',
    'ImagesUsage',
    'InputTokensDetails',
    'IncludeEnum',
    'InferenceOptions',
    'InputAudio',
    'InputFidelity',
    'InputFileContent',
    'InputFileContentParam',
    'InputImageContent',
    'InputImageContentParamAutoParam',
    'InputMessage',
    'InputMessageContentList',
    'InputMessageResource',
    'InputParam',
    'InputTextContent',
    'InputTextContentParam',
    'Invite',
    'ProjectsItem',
    'InviteDeleteResponse',
    'InviteListResponse',
    'InviteProjectGroupBody',
    'InviteRequest',
    'ProjectsItem',
    'ResultsOption0Item',
    'ResultsOption0Item',
    'ItemReferenceParam',
    'ResultsOption0Item',
    'KeyPressAction',
    'ListAssistantsResponse',
    'ListAuditLogsResponse',
    'ListBatchesResponse',
    'ListCertificatesResponse',
    'ListFilesResponse',
    'ListFineTuningCheckpointPermissionResponse',
    'ListFineTuningJobCheckpointsResponse',
    'ListFineTuningJobEventsResponse',
    'ListMessagesResponse',
    'ListModelsResponse',
    'ListPaginatedFineTuningJobsResponse',
    'ListRunStepsResponse',
    'ListRunsResponse',
    'ListVectorStoreFilesResponse',
    'ListVectorStoresResponse',
    'LocalShellCallStatus',
    'LocalShellExecAction',
    'LocalShellToolCall',
    'LocalShellToolCallOutput',
    'LocalShellToolParam',
    'LockedStatus',
    'LogProb',
    'LogProbProperties',
    'MCPApprovalRequest',
    'MCPApprovalResponse',
    'MCPApprovalResponseResource',
    'MCPListTools',
    'MCPListToolsTool',
    'MCPTool',
    'MCPToolApprovalFilter',
    'MCPToolCall',
    'MCPToolCallStatus',
    'MCPToolFilter',
    'Message',
    'ImageFile',
    'MessageContentImageFileObject',
    'ImageUrl',
    'MessageContentImageUrlObject',
    'MessageContentRefusalObject',
    'FileCitation',
    'MessageContentTextAnnotationsFileCitationObject',
    'FilePath',
    'MessageContentTextAnnotationsFilePathObject',
    'MessageContentTextObject',
    'Text',
    'ImageFile',
    'MessageDeltaContentImageFileObject',
    'ImageUrl',
    'MessageDeltaContentImageUrlObject',
    'MessageDeltaContentRefusalObject',
    'FileCitation',
    'MessageDeltaContentTextAnnotationsFileCitationObject',
    'FilePath',
    'MessageDeltaContentTextAnnotationsFilePathObject',
    'MessageDeltaContentTextObject',
    'Text',
    'Delta',
    'MessageDeltaObject',
    'AttachmentsOption0Item',
    'IncompleteDetails',
    'MessageObject',
    'MessageRequestContentTextObject',
    'MessageRole',
    'MessageStatus',
    'AttachmentsOption0Item',
    'Delta',
    'IncompleteDetails',
    'MessageStreamEvent',
    'Metadata',
    'Model',
    'ModelResponseProperties',
    'ImageUrl',
    'ModerationImageURLInput',
    'ModerationTextInput',
    'CodeInterpreter',
    'FileSearch',
    'ModifyAssistantRequest',
    'ToolResources',
    'ModifyCertificateRequest',
    'ModifyMessageRequest',
    'ModifyRunRequest',
    'CodeInterpreter',
    'FileSearch',
    'ModifyThreadRequest',
    'ToolResources',
    'Move',
    'NoiseReductionType',
    'OpenAIFile',
    'OrderEnum',
    'OtherChunkingStrategyResponseParam',
    'OutputAudio',
    'ResultsOption0Item',
    'OutputMessage',
    'OutputTextContent',
    'ParallelToolCalls',
    'PartialImages',
    'PredictionContent',
    'Project',
    'Owner',
    'ProjectApiKey',
    'ProjectApiKeyDeleteResponse',
    'ProjectApiKeyListResponse',
    'ProjectCreateRequest',
    'ProjectGroup',
    'ProjectGroupDeletedResource',
    'ProjectGroupListResource',
    'ProjectListResponse',
    'ProjectRateLimit',
    'ProjectRateLimitListResponse',
    'ProjectRateLimitUpdateRequest',
    'ProjectServiceAccount',
    'ProjectServiceAccountApiKey',
    'ProjectServiceAccountCreateRequest',
    'ProjectServiceAccountCreateResponse',
    'ProjectServiceAccountDeleteResponse',
    'ProjectServiceAccountListResponse',
    'ProjectUpdateRequest',
    'ProjectUser',
    'ProjectUserCreateRequest',
    'ProjectUserDeleteResponse',
    'ProjectUserListResponse',
    'ProjectUserUpdateRequest',
    'Prompt',
    'PublicAssignOrganizationGroupRoleBody',
    'PublicCreateOrganizationRoleBody',
    'PublicRoleListResource',
    'PublicUpdateOrganizationRoleBody',
    'RankerVersionType',
    'RankingOptions',
    'RateLimitsParam',
    'RealtimeAudioFormats',
    'RealtimeBetaClientEventConversationItemCreate',
    'RealtimeBetaClientEventConversationItemDelete',
    'RealtimeBetaClientEventConversationItemRetrieve',
    'RealtimeBetaClientEventConversationItemTruncate',
    'RealtimeBetaClientEventInputAudioBufferAppend',
    'RealtimeBetaClientEventInputAudioBufferClear',
    'RealtimeBetaClientEventInputAudioBufferCommit',
    'RealtimeBetaClientEventOutputAudioBufferClear',
    'RealtimeBetaClientEventResponseCancel',
    'RealtimeBetaClientEventResponseCreate',
    'ToolsItem',
    'ClientSecret',
    'InputAudioTranscription',
    'RealtimeBetaClientEventSessionUpdate',
    'ToolsItem',
    'TracingConfiguration',
    'TurnDetection',
    'InputAudioNoiseReduction',
    'RealtimeBetaClientEventTranscriptionSessionUpdate',
    'TurnDetection',
    'CachedTokensDetails',
    'Error',
    'InputTokenDetails',
    'OutputTokenDetails',
    'RealtimeBetaResponse',
    'StatusDetails',
    'Usage',
    'RealtimeBetaResponseCreateParams',
    'ToolsItem',
    'RealtimeBetaServerEventConversationItemCreated',
    'RealtimeBetaServerEventConversationItemDeleted',
    'RealtimeBetaServerEventConversationItemInputAudioTranscriptionCompleted',
    'RealtimeBetaServerEventConversationItemInputAudioTranscriptionDelta',
    'Error',
    'RealtimeBetaServerEventConversationItemInputAudioTranscriptionFailed',
    'RealtimeBetaServerEventConversationItemInputAudioTranscriptionSegment',
    'RealtimeBetaServerEventConversationItemRetrieved',
    'RealtimeBetaServerEventConversationItemTruncated',
    'Error',
    'RealtimeBetaServerEventError',
    'RealtimeBetaServerEventInputAudioBufferCleared',
    'RealtimeBetaServerEventInputAudioBufferCommitted',
    'RealtimeBetaServerEventInputAudioBufferSpeechStarted',
    'RealtimeBetaServerEventInputAudioBufferSpeechStopped',
    'RealtimeBetaServerEventMCPListToolsCompleted',
    'RealtimeBetaServerEventMCPListToolsFailed',
    'RealtimeBetaServerEventMCPListToolsInProgress',
    'RateLimitsItem',
    'RealtimeBetaServerEventRateLimitsUpdated',
    'RealtimeBetaServerEventResponseAudioDelta',
    'RealtimeBetaServerEventResponseAudioDone',
    'RealtimeBetaServerEventResponseAudioTranscriptDelta',
    'RealtimeBetaServerEventResponseAudioTranscriptDone',
    'Part',
    'RealtimeBetaServerEventResponseContentPartAdded',
    'Part',
    'RealtimeBetaServerEventResponseContentPartDone',
    'CachedTokensDetails',
    'Error',
    'InputTokenDetails',
    'OutputTokenDetails',
    'RealtimeBetaServerEventResponseCreated',
    'StatusDetails',
    'Usage',
    'CachedTokensDetails',
    'Error',
    'InputTokenDetails',
    'OutputTokenDetails',
    'RealtimeBetaServerEventResponseDone',
    'StatusDetails',
    'Usage',
    'RealtimeBetaServerEventResponseFunctionCallArgumentsDelta',
    'RealtimeBetaServerEventResponseFunctionCallArgumentsDone',
    'RealtimeBetaServerEventResponseMCPCallArgumentsDelta',
    'RealtimeBetaServerEventResponseMCPCallArgumentsDone',
    'RealtimeBetaServerEventResponseMCPCallCompleted',
    'RealtimeBetaServerEventResponseMCPCallFailed',
    'RealtimeBetaServerEventResponseMCPCallInProgress',
    'RealtimeBetaServerEventResponseOutputItemAdded',
    'RealtimeBetaServerEventResponseOutputItemDone',
    'RealtimeBetaServerEventResponseTextDelta',
    'RealtimeBetaServerEventResponseTextDone',
    'InputAudioNoiseReduction',
    'RealtimeBetaServerEventSessionCreated',
    'TracingConfiguration',
    'InputAudioNoiseReduction',
    'RealtimeBetaServerEventSessionUpdated',
    'TracingConfiguration',
    'ClientSecret',
    'RealtimeBetaServerEventTranscriptionSessionCreated',
    'TurnDetection',
    'ClientSecret',
    'RealtimeBetaServerEventTranscriptionSessionUpdated',
    'TurnDetection',
    'RealtimeCallCreateRequest',
    'RealtimeCallReferRequest',
    'RealtimeCallRejectRequest',
    'Audio',
    'Output',
    'RealtimeClientEventConversationItemCreate',
    'RealtimeClientEventConversationItemDelete',
    'RealtimeClientEventConversationItemRetrieve',
    'RealtimeClientEventConversationItemTruncate',
    'RealtimeClientEventInputAudioBufferAppend',
    'RealtimeClientEventInputAudioBufferClear',
    'RealtimeClientEventInputAudioBufferCommit',
    'RealtimeClientEventOutputAudioBufferClear',
    'RealtimeClientEventResponseCancel',
    'Audio',
    'Output',
    'RealtimeClientEventResponseCreate',
    'RealtimeClientEventSessionUpdate',
    'InputAudioNoiseReduction',
    'RealtimeClientEventTranscriptionSessionUpdate',
    'TurnDetection',
    'RealtimeConnectParams',
    'RealtimeConversationItemFunctionCall',
    'RealtimeConversationItemFunctionCallOutput',
    'ContentItem',
    'RealtimeConversationItemMessageAssistant',
    'ContentItem',
    'RealtimeConversationItemMessageSystem',
    'ContentItem',
    'RealtimeConversationItemMessageUser',
    'ContentItem',
    'RealtimeConversationItemWithReference',
    'ExpiresAfter',
    'RealtimeCreateClientSecretRequest',
    'RealtimeCreateClientSecretResponse',
    'RealtimeFunctionTool',
    'RealtimeMCPApprovalRequest',
    'RealtimeMCPApprovalResponse',
    'RealtimeMCPListTools',
    'RealtimeMCPProtocolError',
    'RealtimeMCPToolCall',
    'RealtimeMCPToolExecutionError',
    'RealtimeMCPHTTPError',
    'Audio',
    'CachedTokensDetails',
    'Error',
    'InputTokenDetails',
    'Output',
    'OutputTokenDetails',
    'RealtimeResponse',
    'StatusDetails',
    'Usage',
    'Audio',
    'Output',
    'RealtimeResponseCreateParams',
    'Audio',
    'CachedTokensDetails',
    'InputTokenDetails',
    'Output',
    'OutputTokenDetails',
    'StatusDetails',
    'Usage',
    'Conversation',
    'RealtimeServerEventConversationCreated',
    'RealtimeServerEventConversationItemAdded',
    'RealtimeServerEventConversationItemCreated',
    'RealtimeServerEventConversationItemDeleted',
    'RealtimeServerEventConversationItemDone',
    'RealtimeServerEventConversationItemInputAudioTranscriptionCompleted',
    'RealtimeServerEventConversationItemInputAudioTranscriptionDelta',
    'Error',
    'RealtimeServerEventConversationItemInputAudioTranscriptionFailed',
    'RealtimeServerEventConversationItemInputAudioTranscriptionSegment',
    'RealtimeServerEventConversationItemRetrieved',
    'RealtimeServerEventConversationItemTruncated',
    'Error',
    'RealtimeServerEventError',
    'RealtimeServerEventInputAudioBufferCleared',
    'RealtimeServerEventInputAudioBufferCommitted',
    'RealtimeServerEventInputAudioBufferDtmfEventReceived',
    'RealtimeServerEventInputAudioBufferSpeechStarted',
    'RealtimeServerEventInputAudioBufferSpeechStopped',
    'RealtimeServerEventInputAudioBufferTimeoutTriggered',
    'RealtimeServerEventMCPListToolsCompleted',
    'RealtimeServerEventMCPListToolsFailed',
    'RealtimeServerEventMCPListToolsInProgress',
    'RealtimeServerEventOutputAudioBufferCleared',
    'RealtimeServerEventOutputAudioBufferStarted',
    'RealtimeServerEventOutputAudioBufferStopped',
    'RateLimitsItem',
    'RealtimeServerEventRateLimitsUpdated',
    'RealtimeServerEventResponseAudioDelta',
    'RealtimeServerEventResponseAudioDone',
    'RealtimeServerEventResponseAudioTranscriptDelta',
    'RealtimeServerEventResponseAudioTranscriptDone',
    'Part',
    'RealtimeServerEventResponseContentPartAdded',
    'Part',
    'RealtimeServerEventResponseContentPartDone',
    'Audio',
    'CachedTokensDetails',
    'Error',
    'InputTokenDetails',
    'Output',
    'OutputTokenDetails',
    'RealtimeServerEventResponseCreated',
    'StatusDetails',
    'Usage',
    'Audio',
    'CachedTokensDetails',
    'Error',
    'InputTokenDetails',
    'Output',
    'OutputTokenDetails',
    'RealtimeServerEventResponseDone',
    'StatusDetails',
    'Usage',
    'RealtimeServerEventResponseFunctionCallArgumentsDelta',
    'RealtimeServerEventResponseFunctionCallArgumentsDone',
    'RealtimeServerEventResponseMCPCallArgumentsDelta',
    'RealtimeServerEventResponseMCPCallArgumentsDone',
    'RealtimeServerEventResponseMCPCallCompleted',
    'RealtimeServerEventResponseMCPCallFailed',
    'RealtimeServerEventResponseMCPCallInProgress',
    'RealtimeServerEventResponseOutputItemAdded',
    'RealtimeServerEventResponseOutputItemDone',
    'RealtimeServerEventResponseTextDelta',
    'RealtimeServerEventResponseTextDone',
    'RealtimeServerEventSessionCreated',
    'RealtimeServerEventSessionUpdated',
    'ClientSecret',
    'RealtimeServerEventTranscriptionSessionUpdated',
    'TurnDetection',
    'InputAudioNoiseReduction',
    'RealtimeSession',
    'TracingConfiguration',
    'ClientSecret',
    'InputAudioTranscription',
    'RealtimeSessionCreateRequest',
    'ToolsItem',
    'TracingConfiguration',
    'TurnDetection',
    'Audio',
    'Input',
    'NoiseReduction',
    'Output',
    'RealtimeSessionCreateRequestGA',
    'TracingConfiguration',
    'Audio',
    'Input',
    'NoiseReduction',
    'Output',
    'RealtimeSessionCreateResponse',
    'TracingConfiguration',
    'TurnDetection',
    'Audio',
    'ClientSecret',
    'Input',
    'NoiseReduction',
    'Output',
    'RealtimeSessionCreateResponseGA',
    'TracingConfiguration',
    'InputAudioNoiseReduction',
    'RealtimeTranscriptionSessionCreateRequest',
    'TurnDetection',
    'Audio',
    'Input',
    'NoiseReduction',
    'RealtimeTranscriptionSessionCreateRequestGA',
    'ClientSecret',
    'RealtimeTranscriptionSessionCreateResponse',
    'TurnDetection',
    'Audio',
    'Input',
    'NoiseReduction',
    'RealtimeTranscriptionSessionCreateResponseGA',
    'TurnDetection',
    'TokenLimits',
    'RealtimeTurnDetection',
    'Reasoning',
    'ReasoningItem',
    'ReasoningTextContent',
    'RefusalContent',
    'IncompleteDetails',
    'InputTokensDetails',
    'OutputTokensDetails',
    'Response',
    'ResponseAudioDeltaEvent',
    'ResponseAudioDoneEvent',
    'ResponseAudioTranscriptDeltaEvent',
    'ResponseAudioTranscriptDoneEvent',
    'ResponseCodeInterpreterCallCodeDeltaEvent',
    'ResponseCodeInterpreterCallCodeDoneEvent',
    'ResponseCodeInterpreterCallCompletedEvent',
    'ResponseCodeInterpreterCallInProgressEvent',
    'ResponseCodeInterpreterCallInterpretingEvent',
    'ResponseCompletedEvent',
    'ResponseContentPartAddedEvent',
    'ResponseContentPartDoneEvent',
    'ResponseCreatedEvent',
    'ResponseCustomToolCallInputDeltaEvent',
    'ResponseCustomToolCallInputDoneEvent',
    'ResponseError',
    'ResponseErrorCode',
    'ResponseErrorEvent',
    'ResponseFailedEvent',
    'ResponseFileSearchCallCompletedEvent',
    'ResponseFileSearchCallInProgressEvent',
    'ResponseFileSearchCallSearchingEvent',
    'ResponseFormatJsonObject',
    'JsonSchema',
    'ResponseFormatJsonSchema',
    'ResponseFormatJsonSchemaSchema',
    'ResponseFormatText',
    'ResponseFormatTextGrammar',
    'ResponseFormatTextPython',
    'ResponseFunctionCallArgumentsDeltaEvent',
    'ResponseFunctionCallArgumentsDoneEvent',
    'ResponseImageGenCallCompletedEvent',
    'ResponseImageGenCallGeneratingEvent',
    'ResponseImageGenCallInProgressEvent',
    'ResponseImageGenCallPartialImageEvent',
    'ResponseInProgressEvent',
    'ResponseIncompleteEvent',
    'ResponseItemList',
    'ResponseLogProb',
    'TopLogprobsItem',
    'ResponseMCPCallArgumentsDeltaEvent',
    'ResponseMCPCallArgumentsDoneEvent',
    'ResponseMCPCallCompletedEvent',
    'ResponseMCPCallFailedEvent',
    'ResponseMCPCallInProgressEvent',
    'ResponseMCPListToolsCompletedEvent',
    'ResponseMCPListToolsFailedEvent',
    'ResponseMCPListToolsInProgressEvent',
    'ResponseModalities',
    'ResponseOutputItemAddedEvent',
    'ResponseOutputItemDoneEvent',
    'ResponseOutputText',
    'ResponseOutputTextAnnotationAddedEvent',
    'ResponsePromptVariables',
    'ResponseProperties',
    'ResponseQueuedEvent',
    'Part',
    'ResponseReasoningSummaryPartAddedEvent',
    'Part',
    'ResponseReasoningSummaryPartDoneEvent',
    'ResponseReasoningSummaryTextDeltaEvent',
    'ResponseReasoningSummaryTextDoneEvent',
    'ResponseReasoningTextDeltaEvent',
    'ResponseReasoningTextDoneEvent',
    'ResponseRefusalDeltaEvent',
    'ResponseRefusalDoneEvent',
    'ResponseStreamOptions',
    'ResponseTextDeltaEvent',
    'ResponseTextDoneEvent',
    'ResponseTextParam',
    'InputTokensDetails',
    'OutputTokensDetails',
    'ResponseUsage',
    'ResponseWebSearchCallCompletedEvent',
    'ResponseWebSearchCallInProgressEvent',
    'ResponseWebSearchCallSearchingEvent',
    'Role',
    'RoleDeletedResource',
    'RoleListResource',
    'RunCompletionUsage',
    'RunGraderRequest',
    'Errors',
    'Metadata',
    'RunGraderResponse',
    'IncompleteDetails',
    'LastError',
    'RequiredAction',
    'RunObject',
    'SubmitToolOutputs',
    'RunStatus',
    'RunStepCompletionUsage',
    'RunStepDeltaObject',
    'RunStepDeltaObjectDelta',
    'MessageCreation',
    'RunStepDeltaStepDetailsMessageCreationObject',
    'CodeInterpreter',
    'RunStepDeltaStepDetailsToolCallsCodeObject',
    'Image',
    'RunStepDeltaStepDetailsToolCallsCodeOutputImageObject',
    'RunStepDeltaStepDetailsToolCallsCodeOutputLogsObject',
    'RunStepDeltaStepDetailsToolCallsFileSearchObject',
    'Function',
    'RunStepDeltaStepDetailsToolCallsFunctionObject',
    'RunStepDeltaStepDetailsToolCallsObject',
    'MessageCreation',
    'RunStepDetailsMessageCreationObject',
    'CodeInterpreter',
    'RunStepDetailsToolCallsCodeObject',
    'Image',
    'RunStepDetailsToolCallsCodeOutputImageObject',
    'RunStepDetailsToolCallsCodeOutputLogsObject',
    'FileSearch',
    'RunStepDetailsToolCallsFileSearchObject',
    'RunStepDetailsToolCallsFileSearchRankingOptionsObject',
    'ContentItem',
    'RunStepDetailsToolCallsFileSearchResultObject',
    'Function',
    'RunStepDetailsToolCallsFunctionObject',
    'RunStepDetailsToolCallsObject',
    'LastError',
    'RunStepObject',
    'LastError',
    'RunStepStreamEvent',
    'IncompleteDetails',
    'LastError',
    'RequiredAction',
    'RunStreamEvent',
    'SubmitToolOutputs',
    'Function',
    'RunToolCallObject',
    'Screenshot',
    'Scroll',
    'SearchContextSize',
    'SpecificApplyPatchParam',
    'SpecificFunctionShellParam',
    'SpeechAudioDeltaEvent',
    'SpeechAudioDoneEvent',
    'Usage',
    'StaticChunkingStrategy',
    'StaticChunkingStrategyRequestParam',
    'StaticChunkingStrategyResponseParam',
    'StopConfiguration',
    'SubmitToolOutputsRunRequest',
    'ToolOutputsItem',
    'SubmitToolOutputsRunRequestWithoutStream',
    'ToolOutputsItem',
    'Summary',
    'SummaryTextContent',
    'TaskGroupItem',
    'TaskGroupTask',
    'TaskItem',
    'TaskType',
    'TextContent',
    'TextResponseFormatJsonSchema',
    'ThreadItemListResource',
    'ThreadListResource',
    'CodeInterpreter',
    'FileSearch',
    'ThreadObject',
    'ToolResources',
    'ThreadResource',
    'CodeInterpreter',
    'FileSearch',
    'ThreadStreamEvent',
    'ToolResources',
    'ToggleCertificatesRequest',
    'TokenCountsBody',
    'TokenCountsResource',
    'MCPToolApprovalFilter',
    'ToolChoice',
    'ToolChoiceAllowed',
    'ToolChoiceCustom',
    'ToolChoiceFunction',
    'ToolChoiceMCP',
    'ToolChoiceOptions',
    'ToolChoiceTypes',
    'ToolsArray',
    'TopLogProb',
    'LogprobsItem',
    'TranscriptTextDeltaEvent',
    'InputTokenDetails',
    'LogprobsItem',
    'TranscriptTextDoneEvent',
    'TranscriptTextSegmentEvent',
    'TranscriptTextUsageDuration',
    'InputTokenDetails',
    'TranscriptTextUsageTokens',
    'TranscriptionChunkingStrategy',
    'TranscriptionDiarizedSegment',
    'TranscriptionInclude',
    'TranscriptionSegment',
    'TranscriptionWord',
    'TruncationEnum',
    'TruncationObject',
    'Type',
    'UpdateConversationBody',
    'UpdateGroupBody',
    'UpdateVectorStoreFileAttributesRequest',
    'UpdateVectorStoreRequest',
    'UpdateVoiceConsentRequest',
    'Upload',
    'UploadCertificateRequest',
    'UploadPart',
    'UrlAnnotation',
    'UrlAnnotationSource',
    'UrlCitationBody',
    'UsageAudioSpeechesResult',
    'UsageAudioTranscriptionsResult',
    'UsageCodeInterpreterSessionsResult',
    'UsageCompletionsResult',
    'UsageEmbeddingsResult',
    'UsageImagesResult',
    'UsageModerationsResult',
    'UsageResponse',
    'UsageTimeBucket',
    'UsageVectorStoresResult',
    'User',
    'UserDeleteResponse',
    'UserListResource',
    'UserListResponse',
    'UserMessageInputText',
    'UserMessageItem',
    'UserMessageQuotedText',
    'UserRoleAssignment',
    'UserRoleUpdateRequest',
    'VadConfig',
    'ValidateGraderRequest',
    'ValidateGraderResponse',
    'VectorStoreExpirationAfter',
    'VectorStoreFileAttributes',
    'FileCounts',
    'VectorStoreFileBatchObject',
    'DataItem',
    'VectorStoreFileContentResponse',
    'LastError',
    'VectorStoreFileObject',
    'FileCounts',
    'VectorStoreObject',
    'RankingOptions',
    'VectorStoreSearchRequest',
    'VectorStoreSearchResultContentObject',
    'VectorStoreSearchResultItem',
    'VectorStoreSearchResultsPage',
    'VideoContentVariant',
    'VideoListResource',
    'VideoResource',
    'VideoSeconds',
    'VideoSize',
    'VideoStatus',
    'VoiceConsentDeletedResource',
    'VoiceConsentListResource',
    'VoiceConsentResource',
    'VoiceIdsOrCustomVoice',
    'VoiceResource',
    'Wait',
    'WebSearchActionFind',
    'WebSearchActionOpenPage',
    'WebSearchActionSearch',
    'WebSearchSource',
    'WebSearchApproximateLocation',
    'WebSearchContextSize',
    'WebSearchLocation',
    'WebSearchPreviewTool',
    'Filters',
    'WebSearchTool',
    'WebSearchToolCall',
    'Data',
    'WebhookBatchCancelled',
    'Data',
    'WebhookBatchCompleted',
    'Data',
    'WebhookBatchExpired',
    'Data',
    'WebhookBatchFailed',
    'Data',
    'WebhookEvalRunCanceled',
    'Data',
    'WebhookEvalRunFailed',
    'Data',
    'WebhookEvalRunSucceeded',
    'Data',
    'WebhookFineTuningJobCancelled',
    'Data',
    'WebhookFineTuningJobFailed',
    'Data',
    'WebhookFineTuningJobSucceeded',
    'Data',
    'SipHeadersItem',
    'WebhookRealtimeCallIncoming',
    'Data',
    'WebhookResponseCancelled',
    'Data',
    'WebhookResponseCompleted',
    'Data',
    'WebhookResponseFailed',
    'Data',
    'WebhookResponseIncomplete',
    'WidgetMessageItem',
    'WorkflowParam',
    'WorkflowTracingParam',
]

# Re-export all model classes for easy importing and IDE autocompletion
from .active_status import (
    ActiveStatus,
)
from .add_upload_part_request import (
    AddUploadPartRequest,
)
from .admin_api_key import (
    AdminApiKey,
    Owner,
)
from .api_key_list import (
    ApiKeyList,
)
from .apply_patch_call_output_status import (
    ApplyPatchCallOutputStatus,
)
from .apply_patch_call_output_status_param import (
    ApplyPatchCallOutputStatusParam,
)
from .apply_patch_call_status import (
    ApplyPatchCallStatus,
)
from .apply_patch_call_status_param import (
    ApplyPatchCallStatusParam,
)
from .apply_patch_create_file_operation import (
    ApplyPatchCreateFileOperation,
)
from .apply_patch_create_file_operation_param import (
    ApplyPatchCreateFileOperationParam,
)
from .apply_patch_delete_file_operation import (
    ApplyPatchDeleteFileOperation,
)
from .apply_patch_delete_file_operation_param import (
    ApplyPatchDeleteFileOperationParam,
)
from .apply_patch_tool_call import (
    ApplyPatchToolCall,
)
from .apply_patch_tool_call_item_param import (
    ApplyPatchToolCallItemParam,
)
from .apply_patch_tool_call_output import (
    ApplyPatchToolCallOutput,
)
from .apply_patch_tool_call_output_item_param import (
    ApplyPatchToolCallOutputItemParam,
)
from .apply_patch_tool_param import (
    ApplyPatchToolParam,
)
from .apply_patch_update_file_operation import (
    ApplyPatchUpdateFileOperation,
)
from .apply_patch_update_file_operation_param import (
    ApplyPatchUpdateFileOperationParam,
)
from .approximate_location import (
    ApproximateLocation,
)
from .assigned_role_details import (
    AssignedRoleDetails,
)
from .assistant_message_item import (
    AssistantMessageItem,
)
from .assistant_object import (
    AssistantObject,
    CodeInterpreter,
    FileSearch,
    ToolResources,
)
from .assistant_supported_models import (
    AssistantSupportedModels,
)
from .assistant_tools_code import (
    AssistantToolsCode,
)
from .assistant_tools_file_search import (
    AssistantToolsFileSearch,
    FileSearch,
)
from .assistant_tools_file_search_type_only import (
    AssistantToolsFileSearchTypeOnly,
)
from .assistant_tools_function import (
    AssistantToolsFunction,
)
from .assistants_named_tool_choice import (
    AssistantsNamedToolChoice,
    Function,
)
from .attachment import (
    Attachment,
)
from .attachment_type import (
    AttachmentType,
)
from .audio_response_format import (
    AudioResponseFormat,
)
from .audio_transcription import (
    AudioTranscription,
)
from .audit_log import (
    ApiKeyCreated,
    ApiKeyDeleted,
    ApiKeyUpdated,
    AuditLog,
    CertificateCreated,
    CertificateDeleted,
    CertificateUpdated,
    CertificatesActivated,
    CertificatesDeactivated,
    CertificatesItem,
    ChangesRequested,
    CheckpointPermissionCreated,
    CheckpointPermissionDeleted,
    ConfigsItem,
    Data,
    ExternalKeyRegistered,
    ExternalKeyRemoved,
    GroupCreated,
    GroupDeleted,
    GroupUpdated,
    InviteAccepted,
    InviteDeleted,
    InviteSent,
    IpAllowlistConfigActivated,
    IpAllowlistConfigDeactivated,
    IpAllowlistCreated,
    IpAllowlistDeleted,
    IpAllowlistUpdated,
    LoginFailed,
    LogoutFailed,
    OrganizationUpdated,
    Project,
    ProjectArchived,
    ProjectCreated,
    ProjectDeleted,
    ProjectUpdated,
    RateLimitDeleted,
    RateLimitUpdated,
    RoleAssignmentCreated,
    RoleAssignmentDeleted,
    RoleCreated,
    RoleDeleted,
    RoleUpdated,
    ScimDisabled,
    ScimEnabled,
    ServiceAccountCreated,
    ServiceAccountDeleted,
    ServiceAccountUpdated,
    UserAdded,
    UserDeleted,
    UserUpdated,
)
from .audit_log_actor import (
    AuditLogActor,
)
from .audit_log_actor_api_key import (
    AuditLogActorApiKey,
)
from .audit_log_actor_service_account import (
    AuditLogActorServiceAccount,
)
from .audit_log_actor_session import (
    AuditLogActorSession,
)
from .audit_log_actor_user import (
    AuditLogActorUser,
)
from .audit_log_event_type import (
    AuditLogEventType,
)
from .auto_chunking_strategy_request_param import (
    AutoChunkingStrategyRequestParam,
)
from .automatic_thread_titling_param import (
    AutomaticThreadTitlingParam,
)
from .batch import (
    Batch,
    Errors,
    InputTokensDetails,
    OutputTokensDetails,
    Usage,
)
from .batch_error import (
    BatchError,
)
from .batch_file_expiration_after import (
    BatchFileExpirationAfter,
)
from .batch_request_counts import (
    BatchRequestCounts,
)
from .batch_request_input import (
    BatchRequestInput,
)
from .batch_request_output import (
    BatchRequestOutput,
    Error,
    Response,
)
from .certificate import (
    Certificate,
    CertificateDetails,
)
from .chat_completion_allowed_tools import (
    ChatCompletionAllowedTools,
)
from .chat_completion_allowed_tools_choice import (
    ChatCompletionAllowedToolsChoice,
)
from .chat_completion_deleted import (
    ChatCompletionDeleted,
)
from .chat_completion_function_call_option import (
    ChatCompletionFunctionCallOption,
)
from .chat_completion_functions import (
    ChatCompletionFunctions,
)
from .chat_completion_list import (
    ChatCompletionList,
)
from .chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
    Custom,
)
from .chat_completion_message_list import (
    ChatCompletionMessageList,
    DataItem,
)
from .chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from .chat_completion_message_tool_call_chunk import (
    ChatCompletionMessageToolCallChunk,
    Function,
)
from .chat_completion_message_tool_calls import (
    ChatCompletionMessageToolCalls,
)
from .chat_completion_modalities import (
    ChatCompletionModalities,
)
from .chat_completion_named_tool_choice import (
    ChatCompletionNamedToolChoice,
    Function,
)
from .chat_completion_named_tool_choice_custom import (
    ChatCompletionNamedToolChoiceCustom,
    Custom,
)
from .chat_completion_request_assistant_message import (
    Audio,
    ChatCompletionRequestAssistantMessage,
    FunctionCall,
)
from .chat_completion_request_developer_message import (
    ChatCompletionRequestDeveloperMessage,
)
from .chat_completion_request_function_message import (
    ChatCompletionRequestFunctionMessage,
)
from .chat_completion_request_message import (
    Audio,
    FunctionCall,
)
from .chat_completion_request_message_content_part_audio import (
    ChatCompletionRequestMessageContentPartAudio,
    InputAudio,
)
from .chat_completion_request_message_content_part_file import (
    ChatCompletionRequestMessageContentPartFile,
    File,
)
from .chat_completion_request_message_content_part_image import (
    ChatCompletionRequestMessageContentPartImage,
    ImageUrl,
)
from .chat_completion_request_message_content_part_refusal import (
    ChatCompletionRequestMessageContentPartRefusal,
)
from .chat_completion_request_message_content_part_text import (
    ChatCompletionRequestMessageContentPartText,
)
from .chat_completion_request_system_message import (
    ChatCompletionRequestSystemMessage,
)
from .chat_completion_request_tool_message import (
    ChatCompletionRequestToolMessage,
)
from .chat_completion_request_user_message import (
    ChatCompletionRequestUserMessage,
)
from .chat_completion_response_message import (
    AnnotationsItem,
    Audio,
    ChatCompletionResponseMessage,
    FunctionCall,
    UrlCitation,
)
from .chat_completion_role import (
    ChatCompletionRole,
)
from .chat_completion_stream_options import (
    ChatCompletionStreamOptions,
)
from .chat_completion_stream_response_delta import (
    ChatCompletionStreamResponseDelta,
    FunctionCall,
)
from .chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprobsItem,
)
from .chat_completion_tool import (
    ChatCompletionTool,
)
from .chat_model import (
    ChatModel,
)
from .chat_session_automatic_thread_titling import (
    ChatSessionAutomaticThreadTitling,
)
from .chat_session_chatkit_configuration import (
    ChatSessionChatkitConfiguration,
)
from .chat_session_file_upload import (
    ChatSessionFileUpload,
)
from .chat_session_history import (
    ChatSessionHistory,
)
from .chat_session_rate_limits import (
    ChatSessionRateLimits,
)
from .chat_session_resource import (
    ChatSessionResource,
)
from .chat_session_status import (
    ChatSessionStatus,
)
from .chatkit_configuration_param import (
    ChatkitConfigurationParam,
)
from .chatkit_workflow import (
    ChatkitWorkflow,
)
from .chatkit_workflow_tracing import (
    ChatkitWorkflowTracing,
)
from .click_button_type import (
    ClickButtonType,
)
from .click_param import (
    ClickParam,
)
from .client_tool_call_item import (
    ClientToolCallItem,
)
from .client_tool_call_status import (
    ClientToolCallStatus,
)
from .closed_status import (
    ClosedStatus,
)
from .code_interpreter_container_auto import (
    CodeInterpreterContainerAuto,
)
from .code_interpreter_file_output import (
    CodeInterpreterFileOutput,
    FilesItem,
)
from .code_interpreter_output_image import (
    CodeInterpreterOutputImage,
)
from .code_interpreter_output_logs import (
    CodeInterpreterOutputLogs,
)
from .code_interpreter_text_output import (
    CodeInterpreterTextOutput,
)
from .code_interpreter_tool import (
    CodeInterpreterTool,
)
from .code_interpreter_tool_call import (
    CodeInterpreterToolCall,
)
from .compact_resource import (
    CompactResource,
    InputTokensDetails,
    OutputTokensDetails,
)
from .compact_response_method_public_body import (
    CompactResponseMethodPublicBody,
)
from .compaction_body import (
    CompactionBody,
)
from .compaction_summary_item_param import (
    CompactionSummaryItemParam,
)
from .comparison_filter import (
    ComparisonFilter,
)
from .comparison_filter_value_items import (
    ComparisonFilterValueItems,
)
from .complete_upload_request import (
    CompleteUploadRequest,
)
from .completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from .compound_filter import (
    CompoundFilter,
)
from .computer_call_output_item_param import (
    ComputerCallOutputItemParam,
)
from .computer_call_safety_check_param import (
    ComputerCallSafetyCheckParam,
)
from .computer_environment import (
    ComputerEnvironment,
)
from .computer_screenshot_content import (
    ComputerScreenshotContent,
)
from .computer_screenshot_image import (
    ComputerScreenshotImage,
)
from .computer_tool_call import (
    ComputerToolCall,
)
from .computer_tool_call_output import (
    ComputerToolCallOutput,
)
from .computer_tool_call_output_resource import (
    ComputerToolCallOutputResource,
)
from .computer_use_preview_tool import (
    ComputerUsePreviewTool,
)
from .container_file_citation_body import (
    ContainerFileCitationBody,
)
from .container_file_list_resource import (
    ContainerFileListResource,
)
from .container_file_resource import (
    ContainerFileResource,
)
from .container_list_resource import (
    ContainerListResource,
)
from .container_memory_limit import (
    ContainerMemoryLimit,
)
from .container_resource import (
    ContainerResource,
    ExpiresAfter,
)
from .conversation import (
    Conversation,
)
from .conversation_2 import (
    Conversation2,
)
from .conversation_item import (
    ResultsOption0Item,
)
from .conversation_item_list import (
    ConversationItemList,
)
from .conversation_param_2 import (
    ConversationParam2,
)
from .conversation_resource import (
    ConversationResource,
)
from .costs_result import (
    Amount,
    CostsResult,
)
from .create_assistant_request import (
    AutoChunkingStrategy,
    CodeInterpreter,
    CreateAssistantRequest,
    FileSearch,
    Static,
    StaticChunkingStrategy,
    ToolResources,
    VectorStoresItem,
)
from .create_chat_completion_request import (
    Audio,
    CreateChatCompletionRequest,
    UserLocation,
    WebSearchOptions,
)
from .create_chat_completion_response import (
    ChoicesItem,
    CompletionTokensDetails,
    CreateChatCompletionResponse,
    Logprobs,
    PromptTokensDetails,
)
from .create_chat_completion_stream_response import (
    ChoicesItem,
    CompletionTokensDetails,
    CreateChatCompletionStreamResponse,
    Logprobs,
    PromptTokensDetails,
)
from .create_chat_session_body import (
    CreateChatSessionBody,
)
from .create_completion_request import (
    CreateCompletionRequest,
)
from .create_completion_response import (
    ChoicesItem,
    CompletionTokensDetails,
    CreateCompletionResponse,
    Logprobs,
    PromptTokensDetails,
)
from .create_container_body import (
    CreateContainerBody,
    ExpiresAfter,
)
from .create_container_file_body import (
    CreateContainerFileBody,
)
from .create_conversation_body import (
    CreateConversationBody,
)
from .create_embedding_request import (
    CreateEmbeddingRequest,
)
from .create_embedding_response import (
    CreateEmbeddingResponse,
    Usage,
)
from .create_eval_completions_run_data_source import (
    CreateEvalCompletionsRunDataSource,
    ItemReferenceInputMessages,
    SamplingParams,
    TemplateInputMessages,
)
from .create_eval_custom_data_source_config import (
    CreateEvalCustomDataSourceConfig,
)
from .create_eval_jsonl_run_data_source import (
    CreateEvalJsonlRunDataSource,
)
from .create_eval_label_model_grader import (
    CreateEvalLabelModelGrader,
)
from .create_eval_logs_data_source_config import (
    CreateEvalLogsDataSourceConfig,
)
from .create_eval_request import (
    CreateEvalRequest,
)
from .create_eval_responses_run_data_source import (
    ChatMessage,
    CreateEvalResponsesRunDataSource,
    InputMessagesItemReference,
    InputMessagesTemplate,
    SamplingParams,
    Text,
)
from .create_eval_run_request import (
    CreateEvalRunRequest,
)
from .create_eval_stored_completions_data_source_config import (
    CreateEvalStoredCompletionsDataSourceConfig,
)
from .create_file_request import (
    CreateFileRequest,
)
from .create_fine_tuning_checkpoint_permission_request import (
    CreateFineTuningCheckpointPermissionRequest,
)
from .create_fine_tuning_job_request import (
    CreateFineTuningJobRequest,
    Hyperparameters,
    IntegrationsItem,
    Wandb,
)
from .create_group_body import (
    CreateGroupBody,
)
from .create_group_user_body import (
    CreateGroupUserBody,
)
from .create_image_edit_request import (
    CreateImageEditRequest,
)
from .create_image_request import (
    CreateImageRequest,
)
from .create_image_variation_request import (
    CreateImageVariationRequest,
)
from .create_message_request import (
    AttachmentsOption0Item,
    CreateMessageRequest,
)
from .create_model_response_properties import (
    CreateModelResponseProperties,
)
from .create_moderation_request import (
    CreateModerationRequest,
)
from .create_moderation_response import (
    Categories,
    CategoryAppliedInputTypes,
    CategoryScores,
    CreateModerationResponse,
    ResultsItem,
)
from .create_response import (
    CreateResponse,
)
from .create_run_request import (
    CreateRunRequest,
)
from .create_run_request_without_stream import (
    CreateRunRequestWithoutStream,
)
from .create_speech_request import (
    CreateSpeechRequest,
)
from .create_thread_and_run_request import (
    AutoChunkingStrategy,
    CodeInterpreter,
    CreateThreadAndRunRequest,
    FileSearch,
    Static,
    StaticChunkingStrategy,
    ToolResources,
    VectorStoresItem,
)
from .create_thread_and_run_request_without_stream import (
    AutoChunkingStrategy,
    CodeInterpreter,
    CreateThreadAndRunRequestWithoutStream,
    FileSearch,
    Static,
    StaticChunkingStrategy,
    ToolResources,
    VectorStoresItem,
)
from .create_thread_request import (
    AutoChunkingStrategy,
    CodeInterpreter,
    CreateThreadRequest,
    FileSearch,
    Static,
    StaticChunkingStrategy,
    ToolResources,
    VectorStoresItem,
)
from .create_transcription_request import (
    CreateTranscriptionRequest,
)
from .create_transcription_response_diarized_json import (
    CreateTranscriptionResponseDiarizedJson,
)
from .create_transcription_response_json import (
    CreateTranscriptionResponseJson,
    LogprobsItem,
)
from .create_transcription_response_stream_event import (
    InputTokenDetails,
)
from .create_transcription_response_verbose_json import (
    CreateTranscriptionResponseVerboseJson,
)
from .create_translation_request import (
    CreateTranslationRequest,
)
from .create_translation_response_json import (
    CreateTranslationResponseJson,
)
from .create_translation_response_verbose_json import (
    CreateTranslationResponseVerboseJson,
)
from .create_upload_request import (
    CreateUploadRequest,
)
from .create_vector_store_file_batch_request import (
    CreateVectorStoreFileBatchRequest,
)
from .create_vector_store_file_request import (
    CreateVectorStoreFileRequest,
)
from .create_vector_store_request import (
    CreateVectorStoreRequest,
)
from .create_video_body import (
    CreateVideoBody,
)
from .create_video_remix_body import (
    CreateVideoRemixBody,
)
from .create_voice_consent_request import (
    CreateVoiceConsentRequest,
)
from .create_voice_request import (
    CreateVoiceRequest,
)
from .custom_grammar_format_param import (
    CustomGrammarFormatParam,
)
from .custom_text_format_param import (
    CustomTextFormatParam,
)
from .custom_tool_call import (
    CustomToolCall,
)
from .custom_tool_call_output import (
    CustomToolCallOutput,
)
from .custom_tool_chat_completions import (
    Custom,
    CustomToolChatCompletions,
    Grammar,
    GrammarFormat,
    TextFormat,
)
from .custom_tool_param import (
    CustomToolParam,
)
from .delete_assistant_response import (
    DeleteAssistantResponse,
)
from .delete_certificate_response import (
    DeleteCertificateResponse,
)
from .delete_file_response import (
    DeleteFileResponse,
)
from .delete_fine_tuning_checkpoint_permission_response import (
    DeleteFineTuningCheckpointPermissionResponse,
)
from .delete_message_response import (
    DeleteMessageResponse,
)
from .delete_model_response import (
    DeleteModelResponse,
)
from .delete_thread_response import (
    DeleteThreadResponse,
)
from .delete_vector_store_file_response import (
    DeleteVectorStoreFileResponse,
)
from .delete_vector_store_response import (
    DeleteVectorStoreResponse,
)
from .deleted_conversation import (
    DeletedConversation,
)
from .deleted_conversation_resource import (
    DeletedConversationResource,
)
from .deleted_role_assignment_resource import (
    DeletedRoleAssignmentResource,
)
from .deleted_thread_resource import (
    DeletedThreadResource,
)
from .deleted_video_resource import (
    DeletedVideoResource,
)
from .detail_enum import (
    DetailEnum,
)
from .done_event import (
    DoneEvent,
)
from .double_click_action import (
    DoubleClickAction,
)
from .drag import (
    Drag,
)
from .drag_point import (
    DragPoint,
)
from .easy_input_message import (
    EasyInputMessage,
)
from .embedding import (
    Embedding,
)
from .error import (
    Error,
)
from .error_2 import (
    Error2,
)
from .error_event import (
    ErrorEvent,
)
from .error_response import (
    ErrorResponse,
)
from .eval import (
    Eval,
)
from .eval_api_error import (
    EvalApiError,
)
from .eval_custom_data_source_config import (
    EvalCustomDataSourceConfig,
)
from .eval_grader_label_model import (
    EvalGraderLabelModel,
)
from .eval_grader_python import (
    EvalGraderPython,
)
from .eval_grader_score_model import (
    EvalGraderScoreModel,
)
from .eval_grader_string_check import (
    EvalGraderStringCheck,
)
from .eval_grader_text_similarity import (
    EvalGraderTextSimilarity,
)
from .eval_item import (
    EvalItem,
)
from .eval_item_content_array import (
    EvalItemContentArray,
)
from .eval_item_content_output_text import (
    EvalItemContentOutputText,
)
from .eval_item_content_text import (
    EvalItemContentText,
)
from .eval_item_input_image import (
    EvalItemInputImage,
)
from .eval_jsonl_file_content_source import (
    ContentItem,
    EvalJsonlFileContentSource,
)
from .eval_jsonl_file_id_source import (
    EvalJsonlFileIdSource,
)
from .eval_list import (
    EvalList,
)
from .eval_logs_data_source_config import (
    EvalLogsDataSourceConfig,
)
from .eval_responses_source import (
    EvalResponsesSource,
)
from .eval_run import (
    EvalRun,
    PerModelUsageItem,
    PerTestingCriteriaResultsItem,
    ResultCounts,
)
from .eval_run_list import (
    EvalRunList,
)
from .eval_run_output_item import (
    EvalRunOutputItem,
    InputItem,
    OutputItem,
    Sample,
    Usage,
)
from .eval_run_output_item_list import (
    EvalRunOutputItemList,
)
from .eval_run_output_item_result import (
    EvalRunOutputItemResult,
)
from .eval_stored_completions_data_source_config import (
    EvalStoredCompletionsDataSourceConfig,
)
from .eval_stored_completions_source import (
    EvalStoredCompletionsSource,
)
from .expires_after_param import (
    ExpiresAfterParam,
)
from .file_annotation import (
    FileAnnotation,
)
from .file_annotation_source import (
    FileAnnotationSource,
)
from .file_citation_body import (
    FileCitationBody,
)
from .file_expiration_after import (
    FileExpirationAfter,
)
from .file_path import (
    FilePath,
)
from .file_purpose import (
    FilePurpose,
)
from .file_search_ranker import (
    FileSearchRanker,
)
from .file_search_ranking_options import (
    FileSearchRankingOptions,
)
from .file_search_tool import (
    FileSearchTool,
)
from .file_search_tool_call import (
    FileSearchToolCall,
    ResultsOption0Item,
)
from .file_upload_param import (
    FileUploadParam,
)
from .fine_tune_chat_completion_request_assistant_message import (
    Audio,
    FineTuneChatCompletionRequestAssistantMessage,
    FunctionCall,
)
from .fine_tune_chat_request_input import (
    FineTuneChatRequestInput,
)
from .fine_tune_dpo_hyperparameters import (
    FineTuneDPOHyperparameters,
)
from .fine_tune_dpo_method import (
    FineTuneDPOMethod,
)
from .fine_tune_method import (
    FineTuneMethod,
)
from .fine_tune_preference_request_input import (
    FineTunePreferenceRequestInput,
    Input,
)
from .fine_tune_reinforcement_hyperparameters import (
    FineTuneReinforcementHyperparameters,
)
from .fine_tune_reinforcement_method import (
    FineTuneReinforcementMethod,
)
from .fine_tune_reinforcement_request_input import (
    FineTuneReinforcementRequestInput,
)
from .fine_tune_supervised_hyperparameters import (
    FineTuneSupervisedHyperparameters,
)
from .fine_tune_supervised_method import (
    FineTuneSupervisedMethod,
)
from .fine_tuning_checkpoint_permission import (
    FineTuningCheckpointPermission,
)
from .fine_tuning_integration import (
    FineTuningIntegration,
    Wandb,
)
from .fine_tuning_job import (
    Error,
    FineTuningJob,
    Hyperparameters,
)
from .fine_tuning_job_checkpoint import (
    FineTuningJobCheckpoint,
    Metrics,
)
from .fine_tuning_job_event import (
    FineTuningJobEvent,
)
from .function_call_item_status import (
    FunctionCallItemStatus,
)
from .function_call_output_item_param import (
    FunctionCallOutputItemParam,
)
from .function_object import (
    FunctionObject,
)
from .function_parameters import (
    FunctionParameters,
)
from .function_shell_action import (
    FunctionShellAction,
)
from .function_shell_action_param import (
    FunctionShellActionParam,
)
from .function_shell_call import (
    FunctionShellCall,
)
from .function_shell_call_item_param import (
    FunctionShellCallItemParam,
)
from .function_shell_call_item_status import (
    FunctionShellCallItemStatus,
)
from .function_shell_call_output import (
    FunctionShellCallOutput,
)
from .function_shell_call_output_content import (
    FunctionShellCallOutputContent,
)
from .function_shell_call_output_content_param import (
    FunctionShellCallOutputContentParam,
)
from .function_shell_call_output_exit_outcome import (
    FunctionShellCallOutputExitOutcome,
)
from .function_shell_call_output_exit_outcome_param import (
    FunctionShellCallOutputExitOutcomeParam,
)
from .function_shell_call_output_item_param import (
    FunctionShellCallOutputItemParam,
)
from .function_shell_call_output_timeout_outcome import (
    FunctionShellCallOutputTimeoutOutcome,
)
from .function_shell_call_output_timeout_outcome_param import (
    FunctionShellCallOutputTimeoutOutcomeParam,
)
from .function_shell_tool_param import (
    FunctionShellToolParam,
)
from .function_tool import (
    FunctionTool,
)
from .function_tool_call import (
    FunctionToolCall,
)
from .function_tool_call_output import (
    FunctionToolCallOutput,
)
from .function_tool_call_output_resource import (
    FunctionToolCallOutputResource,
)
from .function_tool_call_resource import (
    FunctionToolCallResource,
)
from .grader_label_model import (
    GraderLabelModel,
)
from .grader_multi import (
    GraderMulti,
)
from .grader_python import (
    GraderPython,
)
from .grader_score_model import (
    GraderScoreModel,
    SamplingParams,
)
from .grader_string_check import (
    GraderStringCheck,
)
from .grader_text_similarity import (
    GraderTextSimilarity,
)
from .grammar_syntax1 import (
    GrammarSyntax1,
)
from .group import (
    Group,
)
from .group_deleted_resource import (
    GroupDeletedResource,
)
from .group_list_resource import (
    GroupListResource,
)
from .group_resource_with_success import (
    GroupResourceWithSuccess,
)
from .group_response import (
    GroupResponse,
)
from .group_role_assignment import (
    GroupRoleAssignment,
)
from .group_user_assignment import (
    GroupUserAssignment,
)
from .group_user_deleted_resource import (
    GroupUserDeletedResource,
)
from .history_param import (
    HistoryParam,
)
from .hybrid_search_options import (
    HybridSearchOptions,
)
from .image import (
    Image,
)
from .image_detail import (
    ImageDetail,
)
from .image_edit_completed_event import (
    ImageEditCompletedEvent,
    InputTokensDetails,
)
from .image_edit_partial_image_event import (
    ImageEditPartialImageEvent,
)
from .image_edit_stream_event import (
    InputTokensDetails,
)
from .image_gen_completed_event import (
    ImageGenCompletedEvent,
    InputTokensDetails,
)
from .image_gen_input_usage_details import (
    ImageGenInputUsageDetails,
)
from .image_gen_output_tokens_details import (
    ImageGenOutputTokensDetails,
)
from .image_gen_partial_image_event import (
    ImageGenPartialImageEvent,
)
from .image_gen_stream_event import (
    InputTokensDetails,
)
from .image_gen_tool import (
    ImageGenTool,
    InputImageMask,
)
from .image_gen_tool_call import (
    ImageGenToolCall,
)
from .image_gen_usage import (
    ImageGenUsage,
)
from .images_response import (
    ImagesResponse,
)
from .images_usage import (
    ImagesUsage,
    InputTokensDetails,
)
from .include_enum import (
    IncludeEnum,
)
from .inference_options import (
    InferenceOptions,
)
from .input_audio import (
    InputAudio,
)
from .input_fidelity import (
    InputFidelity,
)
from .input_file_content import (
    InputFileContent,
)
from .input_file_content_param import (
    InputFileContentParam,
)
from .input_image_content import (
    InputImageContent,
)
from .input_image_content_param_auto_param import (
    InputImageContentParamAutoParam,
)
from .input_message import (
    InputMessage,
)
from .input_message_content_list import (
    InputMessageContentList,
)
from .input_message_resource import (
    InputMessageResource,
)
from .input_param import (
    InputParam,
)
from .input_text_content import (
    InputTextContent,
)
from .input_text_content_param import (
    InputTextContentParam,
)
from .invite import (
    Invite,
    ProjectsItem,
)
from .invite_delete_response import (
    InviteDeleteResponse,
)
from .invite_list_response import (
    InviteListResponse,
)
from .invite_project_group_body import (
    InviteProjectGroupBody,
)
from .invite_request import (
    InviteRequest,
    ProjectsItem,
)
from .item import (
    ResultsOption0Item,
)
from .item_field import (
    ResultsOption0Item,
)
from .item_reference_param import (
    ItemReferenceParam,
)
from .item_resource import (
    ResultsOption0Item,
)
from .key_press_action import (
    KeyPressAction,
)
from .list_assistants_response import (
    ListAssistantsResponse,
)
from .list_audit_logs_response import (
    ListAuditLogsResponse,
)
from .list_batches_response import (
    ListBatchesResponse,
)
from .list_certificates_response import (
    ListCertificatesResponse,
)
from .list_files_response import (
    ListFilesResponse,
)
from .list_fine_tuning_checkpoint_permission_response import (
    ListFineTuningCheckpointPermissionResponse,
)
from .list_fine_tuning_job_checkpoints_response import (
    ListFineTuningJobCheckpointsResponse,
)
from .list_fine_tuning_job_events_response import (
    ListFineTuningJobEventsResponse,
)
from .list_messages_response import (
    ListMessagesResponse,
)
from .list_models_response import (
    ListModelsResponse,
)
from .list_paginated_fine_tuning_jobs_response import (
    ListPaginatedFineTuningJobsResponse,
)
from .list_run_steps_response import (
    ListRunStepsResponse,
)
from .list_runs_response import (
    ListRunsResponse,
)
from .list_vector_store_files_response import (
    ListVectorStoreFilesResponse,
)
from .list_vector_stores_response import (
    ListVectorStoresResponse,
)
from .local_shell_call_status import (
    LocalShellCallStatus,
)
from .local_shell_exec_action import (
    LocalShellExecAction,
)
from .local_shell_tool_call import (
    LocalShellToolCall,
)
from .local_shell_tool_call_output import (
    LocalShellToolCallOutput,
)
from .local_shell_tool_param import (
    LocalShellToolParam,
)
from .locked_status import (
    LockedStatus,
)
from .log_prob import (
    LogProb,
)
from .log_prob_properties import (
    LogProbProperties,
)
from .mcp_approval_request import (
    MCPApprovalRequest,
)
from .mcp_approval_response import (
    MCPApprovalResponse,
)
from .mcp_approval_response_resource import (
    MCPApprovalResponseResource,
)
from .mcp_list_tools import (
    MCPListTools,
)
from .mcp_list_tools_tool import (
    MCPListToolsTool,
)
from .mcp_tool import (
    MCPTool,
    MCPToolApprovalFilter,
)
from .mcp_tool_call import (
    MCPToolCall,
)
from .mcp_tool_call_status import (
    MCPToolCallStatus,
)
from .mcp_tool_filter import (
    MCPToolFilter,
)
from .message import (
    Message,
)
from .message_content_image_file_object import (
    ImageFile,
    MessageContentImageFileObject,
)
from .message_content_image_url_object import (
    ImageUrl,
    MessageContentImageUrlObject,
)
from .message_content_refusal_object import (
    MessageContentRefusalObject,
)
from .message_content_text_annotations_file_citation_object import (
    FileCitation,
    MessageContentTextAnnotationsFileCitationObject,
)
from .message_content_text_annotations_file_path_object import (
    FilePath,
    MessageContentTextAnnotationsFilePathObject,
)
from .message_content_text_object import (
    MessageContentTextObject,
    Text,
)
from .message_delta_content_image_file_object import (
    ImageFile,
    MessageDeltaContentImageFileObject,
)
from .message_delta_content_image_url_object import (
    ImageUrl,
    MessageDeltaContentImageUrlObject,
)
from .message_delta_content_refusal_object import (
    MessageDeltaContentRefusalObject,
)
from .message_delta_content_text_annotations_file_citation_object import (
    FileCitation,
    MessageDeltaContentTextAnnotationsFileCitationObject,
)
from .message_delta_content_text_annotations_file_path_object import (
    FilePath,
    MessageDeltaContentTextAnnotationsFilePathObject,
)
from .message_delta_content_text_object import (
    MessageDeltaContentTextObject,
    Text,
)
from .message_delta_object import (
    Delta,
    MessageDeltaObject,
)
from .message_object import (
    AttachmentsOption0Item,
    IncompleteDetails,
    MessageObject,
)
from .message_request_content_text_object import (
    MessageRequestContentTextObject,
)
from .message_role import (
    MessageRole,
)
from .message_status import (
    MessageStatus,
)
from .message_stream_event import (
    AttachmentsOption0Item,
    Delta,
    IncompleteDetails,
    MessageStreamEvent,
)
from .metadata import (
    Metadata,
)
from .model import (
    Model,
)
from .model_response_properties import (
    ModelResponseProperties,
)
from .moderation_image_url_input import (
    ImageUrl,
    ModerationImageURLInput,
)
from .moderation_text_input import (
    ModerationTextInput,
)
from .modify_assistant_request import (
    CodeInterpreter,
    FileSearch,
    ModifyAssistantRequest,
    ToolResources,
)
from .modify_certificate_request import (
    ModifyCertificateRequest,
)
from .modify_message_request import (
    ModifyMessageRequest,
)
from .modify_run_request import (
    ModifyRunRequest,
)
from .modify_thread_request import (
    CodeInterpreter,
    FileSearch,
    ModifyThreadRequest,
    ToolResources,
)
from .move import (
    Move,
)
from .noise_reduction_type import (
    NoiseReductionType,
)
from .open_ai_file import (
    OpenAIFile,
)
from .order_enum import (
    OrderEnum,
)
from .other_chunking_strategy_response_param import (
    OtherChunkingStrategyResponseParam,
)
from .output_audio import (
    OutputAudio,
)
from .output_item import (
    ResultsOption0Item,
)
from .output_message import (
    OutputMessage,
)
from .output_text_content import (
    OutputTextContent,
)
from .parallel_tool_calls import (
    ParallelToolCalls,
)
from .partial_images import (
    PartialImages,
)
from .prediction_content import (
    PredictionContent,
)
from .project import (
    Project,
)
from .project_api_key import (
    Owner,
    ProjectApiKey,
)
from .project_api_key_delete_response import (
    ProjectApiKeyDeleteResponse,
)
from .project_api_key_list_response import (
    ProjectApiKeyListResponse,
)
from .project_create_request import (
    ProjectCreateRequest,
)
from .project_group import (
    ProjectGroup,
)
from .project_group_deleted_resource import (
    ProjectGroupDeletedResource,
)
from .project_group_list_resource import (
    ProjectGroupListResource,
)
from .project_list_response import (
    ProjectListResponse,
)
from .project_rate_limit import (
    ProjectRateLimit,
)
from .project_rate_limit_list_response import (
    ProjectRateLimitListResponse,
)
from .project_rate_limit_update_request import (
    ProjectRateLimitUpdateRequest,
)
from .project_service_account import (
    ProjectServiceAccount,
)
from .project_service_account_api_key import (
    ProjectServiceAccountApiKey,
)
from .project_service_account_create_request import (
    ProjectServiceAccountCreateRequest,
)
from .project_service_account_create_response import (
    ProjectServiceAccountCreateResponse,
)
from .project_service_account_delete_response import (
    ProjectServiceAccountDeleteResponse,
)
from .project_service_account_list_response import (
    ProjectServiceAccountListResponse,
)
from .project_update_request import (
    ProjectUpdateRequest,
)
from .project_user import (
    ProjectUser,
)
from .project_user_create_request import (
    ProjectUserCreateRequest,
)
from .project_user_delete_response import (
    ProjectUserDeleteResponse,
)
from .project_user_list_response import (
    ProjectUserListResponse,
)
from .project_user_update_request import (
    ProjectUserUpdateRequest,
)
from .prompt import (
    Prompt,
)
from .public_assign_organization_group_role_body import (
    PublicAssignOrganizationGroupRoleBody,
)
from .public_create_organization_role_body import (
    PublicCreateOrganizationRoleBody,
)
from .public_role_list_resource import (
    PublicRoleListResource,
)
from .public_update_organization_role_body import (
    PublicUpdateOrganizationRoleBody,
)
from .ranker_version_type import (
    RankerVersionType,
)
from .ranking_options import (
    RankingOptions,
)
from .rate_limits_param import (
    RateLimitsParam,
)
from .realtime_audio_formats import (
    RealtimeAudioFormats,
)
from .realtime_beta_client_event_conversation_item_create import (
    RealtimeBetaClientEventConversationItemCreate,
)
from .realtime_beta_client_event_conversation_item_delete import (
    RealtimeBetaClientEventConversationItemDelete,
)
from .realtime_beta_client_event_conversation_item_retrieve import (
    RealtimeBetaClientEventConversationItemRetrieve,
)
from .realtime_beta_client_event_conversation_item_truncate import (
    RealtimeBetaClientEventConversationItemTruncate,
)
from .realtime_beta_client_event_input_audio_buffer_append import (
    RealtimeBetaClientEventInputAudioBufferAppend,
)
from .realtime_beta_client_event_input_audio_buffer_clear import (
    RealtimeBetaClientEventInputAudioBufferClear,
)
from .realtime_beta_client_event_input_audio_buffer_commit import (
    RealtimeBetaClientEventInputAudioBufferCommit,
)
from .realtime_beta_client_event_output_audio_buffer_clear import (
    RealtimeBetaClientEventOutputAudioBufferClear,
)
from .realtime_beta_client_event_response_cancel import (
    RealtimeBetaClientEventResponseCancel,
)
from .realtime_beta_client_event_response_create import (
    RealtimeBetaClientEventResponseCreate,
    ToolsItem,
)
from .realtime_beta_client_event_session_update import (
    ClientSecret,
    InputAudioTranscription,
    RealtimeBetaClientEventSessionUpdate,
    ToolsItem,
    TracingConfiguration,
    TurnDetection,
)
from .realtime_beta_client_event_transcription_session_update import (
    InputAudioNoiseReduction,
    RealtimeBetaClientEventTranscriptionSessionUpdate,
    TurnDetection,
)
from .realtime_beta_response import (
    CachedTokensDetails,
    Error,
    InputTokenDetails,
    OutputTokenDetails,
    RealtimeBetaResponse,
    StatusDetails,
    Usage,
)
from .realtime_beta_response_create_params import (
    RealtimeBetaResponseCreateParams,
    ToolsItem,
)
from .realtime_beta_server_event_conversation_item_created import (
    RealtimeBetaServerEventConversationItemCreated,
)
from .realtime_beta_server_event_conversation_item_deleted import (
    RealtimeBetaServerEventConversationItemDeleted,
)
from .realtime_beta_server_event_conversation_item_input_audio_transcription_completed import (
    RealtimeBetaServerEventConversationItemInputAudioTranscriptionCompleted,
)
from .realtime_beta_server_event_conversation_item_input_audio_transcription_delta import (
    RealtimeBetaServerEventConversationItemInputAudioTranscriptionDelta,
)
from .realtime_beta_server_event_conversation_item_input_audio_transcription_failed import (
    Error,
    RealtimeBetaServerEventConversationItemInputAudioTranscriptionFailed,
)
from .realtime_beta_server_event_conversation_item_input_audio_transcription_segment import (
    RealtimeBetaServerEventConversationItemInputAudioTranscriptionSegment,
)
from .realtime_beta_server_event_conversation_item_retrieved import (
    RealtimeBetaServerEventConversationItemRetrieved,
)
from .realtime_beta_server_event_conversation_item_truncated import (
    RealtimeBetaServerEventConversationItemTruncated,
)
from .realtime_beta_server_event_error import (
    Error,
    RealtimeBetaServerEventError,
)
from .realtime_beta_server_event_input_audio_buffer_cleared import (
    RealtimeBetaServerEventInputAudioBufferCleared,
)
from .realtime_beta_server_event_input_audio_buffer_committed import (
    RealtimeBetaServerEventInputAudioBufferCommitted,
)
from .realtime_beta_server_event_input_audio_buffer_speech_started import (
    RealtimeBetaServerEventInputAudioBufferSpeechStarted,
)
from .realtime_beta_server_event_input_audio_buffer_speech_stopped import (
    RealtimeBetaServerEventInputAudioBufferSpeechStopped,
)
from .realtime_beta_server_event_mcp_list_tools_completed import (
    RealtimeBetaServerEventMCPListToolsCompleted,
)
from .realtime_beta_server_event_mcp_list_tools_failed import (
    RealtimeBetaServerEventMCPListToolsFailed,
)
from .realtime_beta_server_event_mcp_list_tools_in_progress import (
    RealtimeBetaServerEventMCPListToolsInProgress,
)
from .realtime_beta_server_event_rate_limits_updated import (
    RateLimitsItem,
    RealtimeBetaServerEventRateLimitsUpdated,
)
from .realtime_beta_server_event_response_audio_delta import (
    RealtimeBetaServerEventResponseAudioDelta,
)
from .realtime_beta_server_event_response_audio_done import (
    RealtimeBetaServerEventResponseAudioDone,
)
from .realtime_beta_server_event_response_audio_transcript_delta import (
    RealtimeBetaServerEventResponseAudioTranscriptDelta,
)
from .realtime_beta_server_event_response_audio_transcript_done import (
    RealtimeBetaServerEventResponseAudioTranscriptDone,
)
from .realtime_beta_server_event_response_content_part_added import (
    Part,
    RealtimeBetaServerEventResponseContentPartAdded,
)
from .realtime_beta_server_event_response_content_part_done import (
    Part,
    RealtimeBetaServerEventResponseContentPartDone,
)
from .realtime_beta_server_event_response_created import (
    CachedTokensDetails,
    Error,
    InputTokenDetails,
    OutputTokenDetails,
    RealtimeBetaServerEventResponseCreated,
    StatusDetails,
    Usage,
)
from .realtime_beta_server_event_response_done import (
    CachedTokensDetails,
    Error,
    InputTokenDetails,
    OutputTokenDetails,
    RealtimeBetaServerEventResponseDone,
    StatusDetails,
    Usage,
)
from .realtime_beta_server_event_response_function_call_arguments_delta import (
    RealtimeBetaServerEventResponseFunctionCallArgumentsDelta,
)
from .realtime_beta_server_event_response_function_call_arguments_done import (
    RealtimeBetaServerEventResponseFunctionCallArgumentsDone,
)
from .realtime_beta_server_event_response_mcp_call_arguments_delta import (
    RealtimeBetaServerEventResponseMCPCallArgumentsDelta,
)
from .realtime_beta_server_event_response_mcp_call_arguments_done import (
    RealtimeBetaServerEventResponseMCPCallArgumentsDone,
)
from .realtime_beta_server_event_response_mcp_call_completed import (
    RealtimeBetaServerEventResponseMCPCallCompleted,
)
from .realtime_beta_server_event_response_mcp_call_failed import (
    RealtimeBetaServerEventResponseMCPCallFailed,
)
from .realtime_beta_server_event_response_mcp_call_in_progress import (
    RealtimeBetaServerEventResponseMCPCallInProgress,
)
from .realtime_beta_server_event_response_output_item_added import (
    RealtimeBetaServerEventResponseOutputItemAdded,
)
from .realtime_beta_server_event_response_output_item_done import (
    RealtimeBetaServerEventResponseOutputItemDone,
)
from .realtime_beta_server_event_response_text_delta import (
    RealtimeBetaServerEventResponseTextDelta,
)
from .realtime_beta_server_event_response_text_done import (
    RealtimeBetaServerEventResponseTextDone,
)
from .realtime_beta_server_event_session_created import (
    InputAudioNoiseReduction,
    RealtimeBetaServerEventSessionCreated,
    TracingConfiguration,
)
from .realtime_beta_server_event_session_updated import (
    InputAudioNoiseReduction,
    RealtimeBetaServerEventSessionUpdated,
    TracingConfiguration,
)
from .realtime_beta_server_event_transcription_session_created import (
    ClientSecret,
    RealtimeBetaServerEventTranscriptionSessionCreated,
    TurnDetection,
)
from .realtime_beta_server_event_transcription_session_updated import (
    ClientSecret,
    RealtimeBetaServerEventTranscriptionSessionUpdated,
    TurnDetection,
)
from .realtime_call_create_request import (
    RealtimeCallCreateRequest,
)
from .realtime_call_refer_request import (
    RealtimeCallReferRequest,
)
from .realtime_call_reject_request import (
    RealtimeCallRejectRequest,
)
from .realtime_client_event import (
    Audio,
    Output,
)
from .realtime_client_event_conversation_item_create import (
    RealtimeClientEventConversationItemCreate,
)
from .realtime_client_event_conversation_item_delete import (
    RealtimeClientEventConversationItemDelete,
)
from .realtime_client_event_conversation_item_retrieve import (
    RealtimeClientEventConversationItemRetrieve,
)
from .realtime_client_event_conversation_item_truncate import (
    RealtimeClientEventConversationItemTruncate,
)
from .realtime_client_event_input_audio_buffer_append import (
    RealtimeClientEventInputAudioBufferAppend,
)
from .realtime_client_event_input_audio_buffer_clear import (
    RealtimeClientEventInputAudioBufferClear,
)
from .realtime_client_event_input_audio_buffer_commit import (
    RealtimeClientEventInputAudioBufferCommit,
)
from .realtime_client_event_output_audio_buffer_clear import (
    RealtimeClientEventOutputAudioBufferClear,
)
from .realtime_client_event_response_cancel import (
    RealtimeClientEventResponseCancel,
)
from .realtime_client_event_response_create import (
    Audio,
    Output,
    RealtimeClientEventResponseCreate,
)
from .realtime_client_event_session_update import (
    RealtimeClientEventSessionUpdate,
)
from .realtime_client_event_transcription_session_update import (
    InputAudioNoiseReduction,
    RealtimeClientEventTranscriptionSessionUpdate,
    TurnDetection,
)
from .realtime_connect_params import (
    RealtimeConnectParams,
)
from .realtime_conversation_item_function_call import (
    RealtimeConversationItemFunctionCall,
)
from .realtime_conversation_item_function_call_output import (
    RealtimeConversationItemFunctionCallOutput,
)
from .realtime_conversation_item_message_assistant import (
    ContentItem,
    RealtimeConversationItemMessageAssistant,
)
from .realtime_conversation_item_message_system import (
    ContentItem,
    RealtimeConversationItemMessageSystem,
)
from .realtime_conversation_item_message_user import (
    ContentItem,
    RealtimeConversationItemMessageUser,
)
from .realtime_conversation_item_with_reference import (
    ContentItem,
    RealtimeConversationItemWithReference,
)
from .realtime_create_client_secret_request import (
    ExpiresAfter,
    RealtimeCreateClientSecretRequest,
)
from .realtime_create_client_secret_response import (
    RealtimeCreateClientSecretResponse,
)
from .realtime_function_tool import (
    RealtimeFunctionTool,
)
from .realtime_mcp_approval_request import (
    RealtimeMCPApprovalRequest,
)
from .realtime_mcp_approval_response import (
    RealtimeMCPApprovalResponse,
)
from .realtime_mcp_list_tools import (
    RealtimeMCPListTools,
)
from .realtime_mcp_protocol_error import (
    RealtimeMCPProtocolError,
)
from .realtime_mcp_tool_call import (
    RealtimeMCPToolCall,
)
from .realtime_mcp_tool_execution_error import (
    RealtimeMCPToolExecutionError,
)
from .realtime_mcphttp_error import (
    RealtimeMCPHTTPError,
)
from .realtime_response import (
    Audio,
    CachedTokensDetails,
    Error,
    InputTokenDetails,
    Output,
    OutputTokenDetails,
    RealtimeResponse,
    StatusDetails,
    Usage,
)
from .realtime_response_create_params import (
    Audio,
    Output,
    RealtimeResponseCreateParams,
)
from .realtime_server_event import (
    Audio,
    CachedTokensDetails,
    InputTokenDetails,
    Output,
    OutputTokenDetails,
    StatusDetails,
    Usage,
)
from .realtime_server_event_conversation_created import (
    Conversation,
    RealtimeServerEventConversationCreated,
)
from .realtime_server_event_conversation_item_added import (
    RealtimeServerEventConversationItemAdded,
)
from .realtime_server_event_conversation_item_created import (
    RealtimeServerEventConversationItemCreated,
)
from .realtime_server_event_conversation_item_deleted import (
    RealtimeServerEventConversationItemDeleted,
)
from .realtime_server_event_conversation_item_done import (
    RealtimeServerEventConversationItemDone,
)
from .realtime_server_event_conversation_item_input_audio_transcription_completed import (
    RealtimeServerEventConversationItemInputAudioTranscriptionCompleted,
)
from .realtime_server_event_conversation_item_input_audio_transcription_delta import (
    RealtimeServerEventConversationItemInputAudioTranscriptionDelta,
)
from .realtime_server_event_conversation_item_input_audio_transcription_failed import (
    Error,
    RealtimeServerEventConversationItemInputAudioTranscriptionFailed,
)
from .realtime_server_event_conversation_item_input_audio_transcription_segment import (
    RealtimeServerEventConversationItemInputAudioTranscriptionSegment,
)
from .realtime_server_event_conversation_item_retrieved import (
    RealtimeServerEventConversationItemRetrieved,
)
from .realtime_server_event_conversation_item_truncated import (
    RealtimeServerEventConversationItemTruncated,
)
from .realtime_server_event_error import (
    Error,
    RealtimeServerEventError,
)
from .realtime_server_event_input_audio_buffer_cleared import (
    RealtimeServerEventInputAudioBufferCleared,
)
from .realtime_server_event_input_audio_buffer_committed import (
    RealtimeServerEventInputAudioBufferCommitted,
)
from .realtime_server_event_input_audio_buffer_dtmf_event_received import (
    RealtimeServerEventInputAudioBufferDtmfEventReceived,
)
from .realtime_server_event_input_audio_buffer_speech_started import (
    RealtimeServerEventInputAudioBufferSpeechStarted,
)
from .realtime_server_event_input_audio_buffer_speech_stopped import (
    RealtimeServerEventInputAudioBufferSpeechStopped,
)
from .realtime_server_event_input_audio_buffer_timeout_triggered import (
    RealtimeServerEventInputAudioBufferTimeoutTriggered,
)
from .realtime_server_event_mcp_list_tools_completed import (
    RealtimeServerEventMCPListToolsCompleted,
)
from .realtime_server_event_mcp_list_tools_failed import (
    RealtimeServerEventMCPListToolsFailed,
)
from .realtime_server_event_mcp_list_tools_in_progress import (
    RealtimeServerEventMCPListToolsInProgress,
)
from .realtime_server_event_output_audio_buffer_cleared import (
    RealtimeServerEventOutputAudioBufferCleared,
)
from .realtime_server_event_output_audio_buffer_started import (
    RealtimeServerEventOutputAudioBufferStarted,
)
from .realtime_server_event_output_audio_buffer_stopped import (
    RealtimeServerEventOutputAudioBufferStopped,
)
from .realtime_server_event_rate_limits_updated import (
    RateLimitsItem,
    RealtimeServerEventRateLimitsUpdated,
)
from .realtime_server_event_response_audio_delta import (
    RealtimeServerEventResponseAudioDelta,
)
from .realtime_server_event_response_audio_done import (
    RealtimeServerEventResponseAudioDone,
)
from .realtime_server_event_response_audio_transcript_delta import (
    RealtimeServerEventResponseAudioTranscriptDelta,
)
from .realtime_server_event_response_audio_transcript_done import (
    RealtimeServerEventResponseAudioTranscriptDone,
)
from .realtime_server_event_response_content_part_added import (
    Part,
    RealtimeServerEventResponseContentPartAdded,
)
from .realtime_server_event_response_content_part_done import (
    Part,
    RealtimeServerEventResponseContentPartDone,
)
from .realtime_server_event_response_created import (
    Audio,
    CachedTokensDetails,
    Error,
    InputTokenDetails,
    Output,
    OutputTokenDetails,
    RealtimeServerEventResponseCreated,
    StatusDetails,
    Usage,
)
from .realtime_server_event_response_done import (
    Audio,
    CachedTokensDetails,
    Error,
    InputTokenDetails,
    Output,
    OutputTokenDetails,
    RealtimeServerEventResponseDone,
    StatusDetails,
    Usage,
)
from .realtime_server_event_response_function_call_arguments_delta import (
    RealtimeServerEventResponseFunctionCallArgumentsDelta,
)
from .realtime_server_event_response_function_call_arguments_done import (
    RealtimeServerEventResponseFunctionCallArgumentsDone,
)
from .realtime_server_event_response_mcp_call_arguments_delta import (
    RealtimeServerEventResponseMCPCallArgumentsDelta,
)
from .realtime_server_event_response_mcp_call_arguments_done import (
    RealtimeServerEventResponseMCPCallArgumentsDone,
)
from .realtime_server_event_response_mcp_call_completed import (
    RealtimeServerEventResponseMCPCallCompleted,
)
from .realtime_server_event_response_mcp_call_failed import (
    RealtimeServerEventResponseMCPCallFailed,
)
from .realtime_server_event_response_mcp_call_in_progress import (
    RealtimeServerEventResponseMCPCallInProgress,
)
from .realtime_server_event_response_output_item_added import (
    RealtimeServerEventResponseOutputItemAdded,
)
from .realtime_server_event_response_output_item_done import (
    RealtimeServerEventResponseOutputItemDone,
)
from .realtime_server_event_response_text_delta import (
    RealtimeServerEventResponseTextDelta,
)
from .realtime_server_event_response_text_done import (
    RealtimeServerEventResponseTextDone,
)
from .realtime_server_event_session_created import (
    RealtimeServerEventSessionCreated,
)
from .realtime_server_event_session_updated import (
    RealtimeServerEventSessionUpdated,
)
from .realtime_server_event_transcription_session_updated import (
    ClientSecret,
    RealtimeServerEventTranscriptionSessionUpdated,
    TurnDetection,
)
from .realtime_session import (
    InputAudioNoiseReduction,
    RealtimeSession,
    TracingConfiguration,
)
from .realtime_session_create_request import (
    ClientSecret,
    InputAudioTranscription,
    RealtimeSessionCreateRequest,
    ToolsItem,
    TracingConfiguration,
    TurnDetection,
)
from .realtime_session_create_request_ga import (
    Audio,
    Input,
    NoiseReduction,
    Output,
    RealtimeSessionCreateRequestGA,
    TracingConfiguration,
)
from .realtime_session_create_response import (
    Audio,
    Input,
    NoiseReduction,
    Output,
    RealtimeSessionCreateResponse,
    TracingConfiguration,
    TurnDetection,
)
from .realtime_session_create_response_ga import (
    Audio,
    ClientSecret,
    Input,
    NoiseReduction,
    Output,
    RealtimeSessionCreateResponseGA,
    TracingConfiguration,
)
from .realtime_transcription_session_create_request import (
    InputAudioNoiseReduction,
    RealtimeTranscriptionSessionCreateRequest,
    TurnDetection,
)
from .realtime_transcription_session_create_request_ga import (
    Audio,
    Input,
    NoiseReduction,
    RealtimeTranscriptionSessionCreateRequestGA,
)
from .realtime_transcription_session_create_response import (
    ClientSecret,
    RealtimeTranscriptionSessionCreateResponse,
    TurnDetection,
)
from .realtime_transcription_session_create_response_ga import (
    Audio,
    Input,
    NoiseReduction,
    RealtimeTranscriptionSessionCreateResponseGA,
    TurnDetection,
)
from .realtime_truncation import (
    TokenLimits,
)
from .realtime_turn_detection import (
    RealtimeTurnDetection,
)
from .reasoning import (
    Reasoning,
)
from .reasoning_item import (
    ReasoningItem,
)
from .reasoning_text_content import (
    ReasoningTextContent,
)
from .refusal_content import (
    RefusalContent,
)
from .response import (
    IncompleteDetails,
    InputTokensDetails,
    OutputTokensDetails,
    Response,
)
from .response_audio_delta_event import (
    ResponseAudioDeltaEvent,
)
from .response_audio_done_event import (
    ResponseAudioDoneEvent,
)
from .response_audio_transcript_delta_event import (
    ResponseAudioTranscriptDeltaEvent,
)
from .response_audio_transcript_done_event import (
    ResponseAudioTranscriptDoneEvent,
)
from .response_code_interpreter_call_code_delta_event import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
)
from .response_code_interpreter_call_code_done_event import (
    ResponseCodeInterpreterCallCodeDoneEvent,
)
from .response_code_interpreter_call_completed_event import (
    ResponseCodeInterpreterCallCompletedEvent,
)
from .response_code_interpreter_call_in_progress_event import (
    ResponseCodeInterpreterCallInProgressEvent,
)
from .response_code_interpreter_call_interpreting_event import (
    ResponseCodeInterpreterCallInterpretingEvent,
)
from .response_completed_event import (
    ResponseCompletedEvent,
)
from .response_content_part_added_event import (
    ResponseContentPartAddedEvent,
)
from .response_content_part_done_event import (
    ResponseContentPartDoneEvent,
)
from .response_created_event import (
    ResponseCreatedEvent,
)
from .response_custom_tool_call_input_delta_event import (
    ResponseCustomToolCallInputDeltaEvent,
)
from .response_custom_tool_call_input_done_event import (
    ResponseCustomToolCallInputDoneEvent,
)
from .response_error import (
    ResponseError,
)
from .response_error_code import (
    ResponseErrorCode,
)
from .response_error_event import (
    ResponseErrorEvent,
)
from .response_failed_event import (
    ResponseFailedEvent,
)
from .response_file_search_call_completed_event import (
    ResponseFileSearchCallCompletedEvent,
)
from .response_file_search_call_in_progress_event import (
    ResponseFileSearchCallInProgressEvent,
)
from .response_file_search_call_searching_event import (
    ResponseFileSearchCallSearchingEvent,
)
from .response_format_json_object import (
    ResponseFormatJsonObject,
)
from .response_format_json_schema import (
    JsonSchema,
    ResponseFormatJsonSchema,
)
from .response_format_json_schema_schema import (
    ResponseFormatJsonSchemaSchema,
)
from .response_format_text import (
    ResponseFormatText,
)
from .response_format_text_grammar import (
    ResponseFormatTextGrammar,
)
from .response_format_text_python import (
    ResponseFormatTextPython,
)
from .response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from .response_function_call_arguments_done_event import (
    ResponseFunctionCallArgumentsDoneEvent,
)
from .response_image_gen_call_completed_event import (
    ResponseImageGenCallCompletedEvent,
)
from .response_image_gen_call_generating_event import (
    ResponseImageGenCallGeneratingEvent,
)
from .response_image_gen_call_in_progress_event import (
    ResponseImageGenCallInProgressEvent,
)
from .response_image_gen_call_partial_image_event import (
    ResponseImageGenCallPartialImageEvent,
)
from .response_in_progress_event import (
    ResponseInProgressEvent,
)
from .response_incomplete_event import (
    ResponseIncompleteEvent,
)
from .response_item_list import (
    ResponseItemList,
)
from .response_log_prob import (
    ResponseLogProb,
    TopLogprobsItem,
)
from .response_mcp_call_arguments_delta_event import (
    ResponseMCPCallArgumentsDeltaEvent,
)
from .response_mcp_call_arguments_done_event import (
    ResponseMCPCallArgumentsDoneEvent,
)
from .response_mcp_call_completed_event import (
    ResponseMCPCallCompletedEvent,
)
from .response_mcp_call_failed_event import (
    ResponseMCPCallFailedEvent,
)
from .response_mcp_call_in_progress_event import (
    ResponseMCPCallInProgressEvent,
)
from .response_mcp_list_tools_completed_event import (
    ResponseMCPListToolsCompletedEvent,
)
from .response_mcp_list_tools_failed_event import (
    ResponseMCPListToolsFailedEvent,
)
from .response_mcp_list_tools_in_progress_event import (
    ResponseMCPListToolsInProgressEvent,
)
from .response_modalities import (
    ResponseModalities,
)
from .response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)
from .response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from .response_output_text import (
    ResponseOutputText,
)
from .response_output_text_annotation_added_event import (
    ResponseOutputTextAnnotationAddedEvent,
)
from .response_prompt_variables import (
    ResponsePromptVariables,
)
from .response_properties import (
    ResponseProperties,
)
from .response_queued_event import (
    ResponseQueuedEvent,
)
from .response_reasoning_summary_part_added_event import (
    Part,
    ResponseReasoningSummaryPartAddedEvent,
)
from .response_reasoning_summary_part_done_event import (
    Part,
    ResponseReasoningSummaryPartDoneEvent,
)
from .response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent,
)
from .response_reasoning_summary_text_done_event import (
    ResponseReasoningSummaryTextDoneEvent,
)
from .response_reasoning_text_delta_event import (
    ResponseReasoningTextDeltaEvent,
)
from .response_reasoning_text_done_event import (
    ResponseReasoningTextDoneEvent,
)
from .response_refusal_delta_event import (
    ResponseRefusalDeltaEvent,
)
from .response_refusal_done_event import (
    ResponseRefusalDoneEvent,
)
from .response_stream_options import (
    ResponseStreamOptions,
)
from .response_text_delta_event import (
    ResponseTextDeltaEvent,
)
from .response_text_done_event import (
    ResponseTextDoneEvent,
)
from .response_text_param import (
    ResponseTextParam,
)
from .response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from .response_web_search_call_completed_event import (
    ResponseWebSearchCallCompletedEvent,
)
from .response_web_search_call_in_progress_event import (
    ResponseWebSearchCallInProgressEvent,
)
from .response_web_search_call_searching_event import (
    ResponseWebSearchCallSearchingEvent,
)
from .role import (
    Role,
)
from .role_deleted_resource import (
    RoleDeletedResource,
)
from .role_list_resource import (
    RoleListResource,
)
from .run_completion_usage import (
    RunCompletionUsage,
)
from .run_grader_request import (
    RunGraderRequest,
)
from .run_grader_response import (
    Errors,
    Metadata,
    RunGraderResponse,
)
from .run_object import (
    IncompleteDetails,
    LastError,
    RequiredAction,
    RunObject,
    SubmitToolOutputs,
)
from .run_status import (
    RunStatus,
)
from .run_step_completion_usage import (
    RunStepCompletionUsage,
)
from .run_step_delta_object import (
    RunStepDeltaObject,
)
from .run_step_delta_object_delta import (
    RunStepDeltaObjectDelta,
)
from .run_step_delta_step_details_message_creation_object import (
    MessageCreation,
    RunStepDeltaStepDetailsMessageCreationObject,
)
from .run_step_delta_step_details_tool_calls_code_object import (
    CodeInterpreter,
    RunStepDeltaStepDetailsToolCallsCodeObject,
)
from .run_step_delta_step_details_tool_calls_code_output_image_object import (
    Image,
    RunStepDeltaStepDetailsToolCallsCodeOutputImageObject,
)
from .run_step_delta_step_details_tool_calls_code_output_logs_object import (
    RunStepDeltaStepDetailsToolCallsCodeOutputLogsObject,
)
from .run_step_delta_step_details_tool_calls_file_search_object import (
    RunStepDeltaStepDetailsToolCallsFileSearchObject,
)
from .run_step_delta_step_details_tool_calls_function_object import (
    Function,
    RunStepDeltaStepDetailsToolCallsFunctionObject,
)
from .run_step_delta_step_details_tool_calls_object import (
    RunStepDeltaStepDetailsToolCallsObject,
)
from .run_step_details_message_creation_object import (
    MessageCreation,
    RunStepDetailsMessageCreationObject,
)
from .run_step_details_tool_calls_code_object import (
    CodeInterpreter,
    RunStepDetailsToolCallsCodeObject,
)
from .run_step_details_tool_calls_code_output_image_object import (
    Image,
    RunStepDetailsToolCallsCodeOutputImageObject,
)
from .run_step_details_tool_calls_code_output_logs_object import (
    RunStepDetailsToolCallsCodeOutputLogsObject,
)
from .run_step_details_tool_calls_file_search_object import (
    FileSearch,
    RunStepDetailsToolCallsFileSearchObject,
)
from .run_step_details_tool_calls_file_search_ranking_options_object import (
    RunStepDetailsToolCallsFileSearchRankingOptionsObject,
)
from .run_step_details_tool_calls_file_search_result_object import (
    ContentItem,
    RunStepDetailsToolCallsFileSearchResultObject,
)
from .run_step_details_tool_calls_function_object import (
    Function,
    RunStepDetailsToolCallsFunctionObject,
)
from .run_step_details_tool_calls_object import (
    RunStepDetailsToolCallsObject,
)
from .run_step_object import (
    LastError,
    RunStepObject,
)
from .run_step_stream_event import (
    LastError,
    RunStepStreamEvent,
)
from .run_stream_event import (
    IncompleteDetails,
    LastError,
    RequiredAction,
    RunStreamEvent,
    SubmitToolOutputs,
)
from .run_tool_call_object import (
    Function,
    RunToolCallObject,
)
from .screenshot import (
    Screenshot,
)
from .scroll import (
    Scroll,
)
from .search_context_size import (
    SearchContextSize,
)
from .specific_apply_patch_param import (
    SpecificApplyPatchParam,
)
from .specific_function_shell_param import (
    SpecificFunctionShellParam,
)
from .speech_audio_delta_event import (
    SpeechAudioDeltaEvent,
)
from .speech_audio_done_event import (
    SpeechAudioDoneEvent,
    Usage,
)
from .static_chunking_strategy import (
    StaticChunkingStrategy,
)
from .static_chunking_strategy_request_param import (
    StaticChunkingStrategyRequestParam,
)
from .static_chunking_strategy_response_param import (
    StaticChunkingStrategyResponseParam,
)
from .stop_configuration import (
    StopConfiguration,
)
from .submit_tool_outputs_run_request import (
    SubmitToolOutputsRunRequest,
    ToolOutputsItem,
)
from .submit_tool_outputs_run_request_without_stream import (
    SubmitToolOutputsRunRequestWithoutStream,
    ToolOutputsItem,
)
from .summary import (
    Summary,
)
from .summary_text_content import (
    SummaryTextContent,
)
from .task_group_item import (
    TaskGroupItem,
)
from .task_group_task import (
    TaskGroupTask,
)
from .task_item import (
    TaskItem,
)
from .task_type import (
    TaskType,
)
from .text_content import (
    TextContent,
)
from .text_response_format_json_schema import (
    TextResponseFormatJsonSchema,
)
from .thread_item_list_resource import (
    ThreadItemListResource,
)
from .thread_list_resource import (
    ThreadListResource,
)
from .thread_object import (
    CodeInterpreter,
    FileSearch,
    ThreadObject,
    ToolResources,
)
from .thread_resource import (
    ThreadResource,
)
from .thread_stream_event import (
    CodeInterpreter,
    FileSearch,
    ThreadStreamEvent,
    ToolResources,
)
from .toggle_certificates_request import (
    ToggleCertificatesRequest,
)
from .token_counts_body import (
    TokenCountsBody,
)
from .token_counts_resource import (
    TokenCountsResource,
)
from .tool import (
    MCPToolApprovalFilter,
)
from .tool_choice import (
    ToolChoice,
)
from .tool_choice_allowed import (
    ToolChoiceAllowed,
)
from .tool_choice_custom import (
    ToolChoiceCustom,
)
from .tool_choice_function import (
    ToolChoiceFunction,
)
from .tool_choice_mcp import (
    ToolChoiceMCP,
)
from .tool_choice_options import (
    ToolChoiceOptions,
)
from .tool_choice_types import (
    ToolChoiceTypes,
)
from .tools_array import (
    ToolsArray,
)
from .top_log_prob import (
    TopLogProb,
)
from .transcript_text_delta_event import (
    LogprobsItem,
    TranscriptTextDeltaEvent,
)
from .transcript_text_done_event import (
    InputTokenDetails,
    LogprobsItem,
    TranscriptTextDoneEvent,
)
from .transcript_text_segment_event import (
    TranscriptTextSegmentEvent,
)
from .transcript_text_usage_duration import (
    TranscriptTextUsageDuration,
)
from .transcript_text_usage_tokens import (
    InputTokenDetails,
    TranscriptTextUsageTokens,
)
from .transcription_chunking_strategy import (
    TranscriptionChunkingStrategy,
)
from .transcription_diarized_segment import (
    TranscriptionDiarizedSegment,
)
from .transcription_include import (
    TranscriptionInclude,
)
from .transcription_segment import (
    TranscriptionSegment,
)
from .transcription_word import (
    TranscriptionWord,
)
from .truncation_enum import (
    TruncationEnum,
)
from .truncation_object import (
    TruncationObject,
)
from .type import (
    Type,
)
from .update_conversation_body import (
    UpdateConversationBody,
)
from .update_group_body import (
    UpdateGroupBody,
)
from .update_vector_store_file_attributes_request import (
    UpdateVectorStoreFileAttributesRequest,
)
from .update_vector_store_request import (
    UpdateVectorStoreRequest,
)
from .update_voice_consent_request import (
    UpdateVoiceConsentRequest,
)
from .upload import (
    Upload,
)
from .upload_certificate_request import (
    UploadCertificateRequest,
)
from .upload_part import (
    UploadPart,
)
from .url_annotation import (
    UrlAnnotation,
)
from .url_annotation_source import (
    UrlAnnotationSource,
)
from .url_citation_body import (
    UrlCitationBody,
)
from .usage_audio_speeches_result import (
    UsageAudioSpeechesResult,
)
from .usage_audio_transcriptions_result import (
    UsageAudioTranscriptionsResult,
)
from .usage_code_interpreter_sessions_result import (
    UsageCodeInterpreterSessionsResult,
)
from .usage_completions_result import (
    UsageCompletionsResult,
)
from .usage_embeddings_result import (
    UsageEmbeddingsResult,
)
from .usage_images_result import (
    UsageImagesResult,
)
from .usage_moderations_result import (
    UsageModerationsResult,
)
from .usage_response import (
    UsageResponse,
)
from .usage_time_bucket import (
    UsageTimeBucket,
)
from .usage_vector_stores_result import (
    UsageVectorStoresResult,
)
from .user import (
    User,
)
from .user_delete_response import (
    UserDeleteResponse,
)
from .user_list_resource import (
    UserListResource,
)
from .user_list_response import (
    UserListResponse,
)
from .user_message_input_text import (
    UserMessageInputText,
)
from .user_message_item import (
    UserMessageItem,
)
from .user_message_quoted_text import (
    UserMessageQuotedText,
)
from .user_role_assignment import (
    UserRoleAssignment,
)
from .user_role_update_request import (
    UserRoleUpdateRequest,
)
from .vad_config import (
    VadConfig,
)
from .validate_grader_request import (
    ValidateGraderRequest,
)
from .validate_grader_response import (
    ValidateGraderResponse,
)
from .vector_store_expiration_after import (
    VectorStoreExpirationAfter,
)
from .vector_store_file_attributes import (
    VectorStoreFileAttributes,
)
from .vector_store_file_batch_object import (
    FileCounts,
    VectorStoreFileBatchObject,
)
from .vector_store_file_content_response import (
    DataItem,
    VectorStoreFileContentResponse,
)
from .vector_store_file_object import (
    LastError,
    VectorStoreFileObject,
)
from .vector_store_object import (
    FileCounts,
    VectorStoreObject,
)
from .vector_store_search_request import (
    RankingOptions,
    VectorStoreSearchRequest,
)
from .vector_store_search_result_content_object import (
    VectorStoreSearchResultContentObject,
)
from .vector_store_search_result_item import (
    VectorStoreSearchResultItem,
)
from .vector_store_search_results_page import (
    VectorStoreSearchResultsPage,
)
from .video_content_variant import (
    VideoContentVariant,
)
from .video_list_resource import (
    VideoListResource,
)
from .video_resource import (
    VideoResource,
)
from .video_seconds import (
    VideoSeconds,
)
from .video_size import (
    VideoSize,
)
from .video_status import (
    VideoStatus,
)
from .voice_consent_deleted_resource import (
    VoiceConsentDeletedResource,
)
from .voice_consent_list_resource import (
    VoiceConsentListResource,
)
from .voice_consent_resource import (
    VoiceConsentResource,
)
from .voice_ids_or_custom_voice import (
    VoiceIdsOrCustomVoice,
)
from .voice_resource import (
    VoiceResource,
)
from .wait import (
    Wait,
)
from .web_search_action_find import (
    WebSearchActionFind,
)
from .web_search_action_open_page import (
    WebSearchActionOpenPage,
)
from .web_search_action_search import (
    WebSearchActionSearch,
    WebSearchSource,
)
from .web_search_approximate_location import (
    WebSearchApproximateLocation,
)
from .web_search_context_size import (
    WebSearchContextSize,
)
from .web_search_location import (
    WebSearchLocation,
)
from .web_search_preview_tool import (
    WebSearchPreviewTool,
)
from .web_search_tool import (
    Filters,
    WebSearchTool,
)
from .web_search_tool_call import (
    WebSearchToolCall,
)
from .webhook_batch_cancelled import (
    Data,
    WebhookBatchCancelled,
)
from .webhook_batch_completed import (
    Data,
    WebhookBatchCompleted,
)
from .webhook_batch_expired import (
    Data,
    WebhookBatchExpired,
)
from .webhook_batch_failed import (
    Data,
    WebhookBatchFailed,
)
from .webhook_eval_run_canceled import (
    Data,
    WebhookEvalRunCanceled,
)
from .webhook_eval_run_failed import (
    Data,
    WebhookEvalRunFailed,
)
from .webhook_eval_run_succeeded import (
    Data,
    WebhookEvalRunSucceeded,
)
from .webhook_fine_tuning_job_cancelled import (
    Data,
    WebhookFineTuningJobCancelled,
)
from .webhook_fine_tuning_job_failed import (
    Data,
    WebhookFineTuningJobFailed,
)
from .webhook_fine_tuning_job_succeeded import (
    Data,
    WebhookFineTuningJobSucceeded,
)
from .webhook_realtime_call_incoming import (
    Data,
    SipHeadersItem,
    WebhookRealtimeCallIncoming,
)
from .webhook_response_cancelled import (
    Data,
    WebhookResponseCancelled,
)
from .webhook_response_completed import (
    Data,
    WebhookResponseCompleted,
)
from .webhook_response_failed import (
    Data,
    WebhookResponseFailed,
)
from .webhook_response_incomplete import (
    Data,
    WebhookResponseIncomplete,
)
from .widget_message_item import (
    WidgetMessageItem,
)
from .workflow_param import (
    WorkflowParam,
)
from .workflow_tracing_param import (
    WorkflowTracingParam,
)