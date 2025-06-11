package config

import (
	"maistro/models"
	"maistro/util"

	"github.com/google/uuid"
)

var DefaultPrimaryProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000001"),
	Name:        "Primary (Default)",
	Type:        int(models.ModelProfileTypePrimary),
	Description: util.StrPtr("Primary model profile for general chat and reasoning."),
	ModelName:   "qwen3:30b-a3b",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(10216), // 10216
		RepeatLastN:   util.IntPtr(-1),
		RepeatPenalty: util.Float32Ptr(1.6),
		Temperature:   util.Float32Ptr(0.9),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(70),
		TopP:          util.Float32Ptr(0.7),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "You are a helpful AI assistant.",
}

var DefaultSummarizationProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000002"),
	Name:        "Summarization (Default)",
	Type:        int(models.ModelProfileTypePrimarySummary),
	Description: util.StrPtr("Default profile for conversation summarization."),
	ModelName:   "cogito:8b",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(10216),
		RepeatLastN:   util.IntPtr(-1),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.3),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Summarize the conversation so far in a concise paragraph. Include key points and conclusions, but omit redundant details.",
}

var DefaultMasterSummaryProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000003"),
	Name:        "Master Summary (Default)",
	Type:        int(models.ModelProfileTypeMasterSummary),
	Description: util.StrPtr("Profile for generating master summaries."),
	ModelName:   "phi4-reasoning:plus",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.3),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Create a comprehensive summary of the conversation, giving most weight to the most recent points and less to older information.",
}

var DefaultBriefSummaryProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000004"),
	Name:        "Brief Summary (Default)",
	Type:        int(models.ModelProfileTypeBriefSummary),
	Description: util.StrPtr("Profile for generating brief summaries."),
	ModelName:   "phi4-reasoning:plus",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.2),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Create a very concise summary of these short messages. Focus only on essential information and be extremely brief.",
}

var DefaultKeyPointsProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000005"),
	Name:        "Key Points (Default)",
	Type:        int(models.ModelProfileTypeKeyPoints),
	Description: util.StrPtr("Profile for extracting key points from messages."),
	ModelName:   "phi4-reasoning:plus",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.2),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Extract and list the key points from these detailed messages. Identify the main ideas and important details, organizing them in a clear structure.",
}

var DefaultSelfCritiqueProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000006"),
	Name:        "Self Critique (Default)",
	Type:        int(models.ModelProfileTypeSelfCritique),
	Description: util.StrPtr("Profile for self-critique and response evaluation."),
	ModelName:   "mistral:latest",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.4),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "You are an expert critique assistant. Your task is to analyze the following AI response and identify:" +
		"\n1. Factual inaccuracies or potential errors" +
		"\n2. Areas where clarity could be improved" +
		"\n3. Opportunities to make the response more helpful or comprehensive" +
		"\n4. Any redundancies or unnecessary content" +
		"\nBe concise and focus on actionable feedback that can improve the response.",
}

var DefaultImprovementProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000007"),
	Name:        "Improvement (Default)",
	Type:        int(models.ModelProfileTypeImprovement),
	Description: util.StrPtr("Profile for improving and refining responses."),
	ModelName:   "mistral:latest",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.4),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Your task is to improve the original AI response based on the critique provided. " +
		"Maintain the overall structure and intent of the original response, but address the issues identified in the critique. " +
		"The improved response should be clear, accurate, concise, and directly answer the user's original query.",
}

var DefaultMemoryRetrievalProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000008"),
	Name:        "Memory Retrieval (Default)",
	Type:        int(models.ModelProfileTypeMemoryRetrieval),
	Description: util.StrPtr("Profile for retrieving and summarizing memory/context."),
	ModelName:   "cogito:8b",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.2),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Retrieve relevant information from memory and present it concisely.",
}

var DefaultAnalysisProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000009"),
	Name:        "Analysis (Default)",
	Type:        int(models.ModelProfileTypeAnalysis),
	Description: util.StrPtr("Profile for analyzing and synthesizing information."),
	ModelName:   "phi4-reasoning:plus",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.2),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Review the provided information and analyze it for key insights. " +
		"Identify trends, patterns, and significant details that can inform future actions or decisions. " +
		"Present your analysis in a clear and structured format." +
		"Ensure to highlight any critical insights that may impact decision-making.",
}

var DefaultResearchTaskProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000010"),
	Name:        "Research Task (Default)",
	Type:        int(models.ModelProfileTypeResearchTask),
	Description: util.StrPtr("Profile for conducting research tasks."),
	ModelName:   "phi4-reasoning:plus",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.7),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "You are a research report writer. You have been provided with findings for sub-topics of a larger research request. " +
		"Combine these findings into a coherent, well-structured report that directly addresses the original user request. " +
		"Start with a brief executive summary, then elaborate on the findings for each sub-question. " +
		"If some sub-questions had errors or insufficient info, acknowledge that in your report. " +
		"Format the report with proper sections, highlighting key points. " +
		"Do not invent information not present in the input.",
}

var DefaultResearchPlanProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000011"),
	Name:        "Research Plan (Default)",
	Type:        int(models.ModelProfileTypeResearchPlan),
	Description: util.StrPtr("Profile for creating research plans."),
	ModelName:   "phi4-reasoning:plus",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.7),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "You are a research planning assistant. Analyze the following user request. " +
		"1. Clarify the core intent and scope. " +
		"2. Break down the request into 3-5 key research questions or sub-topics. " +
		"3. For each sub-topic, suggest 1-3 initial search engine query keywords. ",
}

var DefaultResearchConsolidationProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000012"),
	Name:        "Research Consolidation (Default)",
	Type:        int(models.ModelProfileTypeResearchConsolidation),
	Description: util.StrPtr("Profile for consolidating research findings."),
	ModelName:   "phi4-reasoning:plus",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.7),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "You are a research report writer. You have been provided with findings for sub-topics of a larger research request. " +
		"Combine these findings into a coherent, well-structured report that directly addresses the original user request. " +
		"Start with a brief executive summary, then elaborate on the findings for each sub-question. " +
		"If some sub-questions had errors or insufficient info, acknowledge that in your report. " +
		"Format the report with proper sections, highlighting key points. " +
		"Do not invent information not present in the input.",
}

var DefaultResearchAnalysisProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000013"),
	Name:        "Research Analysis (Default)",
	Type:        int(models.ModelProfileTypeResearchAnalysis),
	Description: util.StrPtr("Profile for analyzing research findings."),
	ModelName:   "phi4-reasoning:plus",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.7),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "You are a research analyst. Based ONLY on the provided text snippets, answer the following research question concisely. " +
		"Synthesize the information and extract key findings. If the text doesn't answer the question, say so explicitly. " +
		"Include references to the sources in your answer when appropriate. ",
}

var DefaultEmbeddingProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000014"),
	Name:        "Embedding (Default)",
	Type:        int(models.ModelProfileTypeEmbedding),
	Description: util.StrPtr("Profile for generating embeddings."),
	ModelName:   "nomic-embed-text:latest",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.0),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Generate a vector embedding for the provided text. The embedding should be a fixed-size vector of 768 dimensions.",
}

var DefaultFormattingProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000015"),
	Name:        "Formatting (Default)",
	Type:        int(models.ModelProfileTypeFormatting),
	Description: util.StrPtr("Profile for formatting text."),
	ModelName:   "cogito:8b",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.0),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Format the provided text according to the specified style. Ensure that the formatting is consistent and adheres to the guidelines.",
}

var DefaultImageGenerationProfile = models.ModelProfile{
	ID:          uuid.MustParse("00000000-0000-0000-0000-000000000016"),
	Name:        "Image Generation (Default)",
	Type:        int(models.ModelProfileTypeImageGeneration),
	Description: util.StrPtr("Profile for generating images based on text prompts."),
	ModelName:   "dalle2:latest",
	Parameters: models.ModelParameters{
		NumCtx:        util.IntPtr(2048),
		RepeatLastN:   util.IntPtr(64),
		RepeatPenalty: util.Float32Ptr(1.1),
		Temperature:   util.Float32Ptr(0.0),
		Seed:          util.IntPtr(0),
		Stop:          []string{},
		NumPredict:    util.IntPtr(-1),
		TopK:          util.IntPtr(40),
		TopP:          util.Float32Ptr(0.9),
		MinP:          util.Float32Ptr(0.0),
	},
	SystemPrompt: "Generate an image based on the provided text prompt. The image should be high quality and relevant to the prompt.",
}

var DefaultModelProfiles = []models.ModelProfile{
	DefaultPrimaryProfile,
	DefaultSummarizationProfile,
	DefaultMasterSummaryProfile,
	DefaultBriefSummaryProfile,
	DefaultKeyPointsProfile,
	DefaultSelfCritiqueProfile,
	DefaultImprovementProfile,
	DefaultMemoryRetrievalProfile,
	DefaultAnalysisProfile,
	DefaultResearchTaskProfile,
	DefaultResearchPlanProfile,
	DefaultResearchConsolidationProfile,
	DefaultResearchAnalysisProfile,
	DefaultEmbeddingProfile,
	DefaultFormattingProfile,
	DefaultImageGenerationProfile,
}
