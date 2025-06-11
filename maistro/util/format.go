package util

import (
	"reflect"
	"regexp"
	"strings"
)

func RemoveThinkTags(input string) string {
	// Remove <think> tags and their content, including newlines
	thinkTagPattern := regexp.MustCompile(`(?s)<think>.*?</think>`) // (?s) makes . match newlines
	return thinkTagPattern.ReplaceAllString(input, "")
}

// SanitizeText removes HTML/XML-like tags, markdown, escape characters, and trims excessive whitespace
func SanitizeText(input string) string {
	// Remove <...> tags and their content (e.g., <think>...</think>)
	tagPattern := regexp.MustCompile(`(?s)<[^>]+>.*?</[^>]+>`)
	withoutTags := tagPattern.ReplaceAllString(input, " ")
	// Remove any remaining standalone tags (e.g., <br>, <hr>, etc.)
	standaloneTagPattern := regexp.MustCompile(`<[^>]+>`)
	withoutStandaloneTags := standaloneTagPattern.ReplaceAllString(withoutTags, " ")

	// Remove markdown syntax (**, __, *, _, `, #, >, ---)
	markdownPattern := regexp.MustCompile(`(\*\*|__|\*|_|` + "`" + `(?s)|#+|>+|---|\[.*?\]\(.*?\))`)
	withoutMarkdown := markdownPattern.ReplaceAllString(withoutStandaloneTags, " ")

	// Remove escape characters (newlines, tabs, carriage returns)
	escapePattern := regexp.MustCompile(`[\n\r\t]`)
	withoutEscapes := escapePattern.ReplaceAllString(withoutMarkdown, " ")

	// Replace multiple whitespace with a single space
	spacePattern := regexp.MustCompile(`\s+`)
	cleaned := spacePattern.ReplaceAllString(withoutEscapes, " ")
	return strings.TrimSpace(cleaned)
}

func StructForEach(s any, fn func(fieldName string, fieldValue any)) {
	val := reflect.ValueOf(s)
	if val.Kind() != reflect.Struct {
		return // Not a struct
	}

	for i := range val.NumField() {
		field := val.Field(i)
		fieldType := val.Type().Field(i)
		fn(fieldType.Name, field.Interface())
	}
}

func StrPtr(s string) *string       { return &s }
func IntPtr(i int) *int             { return &i }
func Float32Ptr(f float32) *float32 { return &f }
func BoolPtr(b bool) *bool          { return &b }
