package util

import (
	"path/filepath"
	"runtime"

	"maps"

	"github.com/sirupsen/logrus"
)

// HandleFatalError is a helper function that logs the error and panics
func HandleFatalError(err error, additionalFields ...logrus.Fields) {
	HandleFatalErrorAtCallLevel(err, 2, additionalFields...)
}

// HandleFatalErrorAtCallLevel is a helper function that logs the error and panics at a specific call level
func HandleFatalErrorAtCallLevel(err error, lvl int, additionalFields ...logrus.Fields) {
	if err == nil {
		return
	}

	_, file, line, _ := runtime.Caller(lvl) // Get the caller of this function

	// Create base fields
	logFields := logrus.Fields{
		"file": filepath.Join(filepath.Base(filepath.Dir(file)), filepath.Base(file)),
		"line": line,
	}

	// Merge any additional fields
	if len(additionalFields) > 0 {
		maps.Copy(logFields, additionalFields[0])
	}

	logrus.WithFields(logFields).Fatal(err)
	panic(err) // Ensure we panic after logging
}

// handleError is a helper function that logs the error and returns a fiber error
// to standardize error handling across API endpoints
func HandleError(err error, additionalFields ...logrus.Fields) error {
	return HandleErrorAtCallLevel(err, 2, additionalFields...)
}

// handleError is a helper function that logs the error and returns a fiber error
// to standardize error handling across API endpoints
func HandleErrorAtCallLevel(err error, lvl int, additionalFields ...logrus.Fields) error {
	if err == nil {
		return nil
	}

	_, file, line, _ := runtime.Caller(lvl) // Get the caller of this function

	// Create base fields
	logFields := logrus.Fields{
		"file": filepath.Join(filepath.Base(filepath.Dir(file)), filepath.Base(file)),
		"line": line,
	}

	// Merge any additional fields
	if len(additionalFields) > 0 {
		maps.Copy(logFields, additionalFields[0])
	}

	logrus.WithFields(logFields).Error(err)
	return err
}

// LogWarning is a helper function that logs a warning
func LogWarning(msg string, additionalFields ...logrus.Fields) {
	LogWarningAtCallLevel(msg, 2, additionalFields...)
}

// LogWarningAtCallLevel is a helper function that logs a warning at a specific call level
func LogWarningAtCallLevel(msg string, lvl int, additionalFields ...logrus.Fields) {
	_, file, line, _ := runtime.Caller(lvl) // Get the caller of this function

	// Create base fields
	logFields := logrus.Fields{
		"file": filepath.Join(filepath.Base(filepath.Dir(file)), filepath.Base(file)),
		"line": line,
	}

	// Merge any additional fields
	if len(additionalFields) > 0 {
		maps.Copy(logFields, additionalFields[0])
	}

	logrus.WithFields(logFields).Warn(msg)
}

// LogInfo is a helper function that logs an info message
func LogInfo(msg string, additionalFields ...logrus.Fields) {
	LogInfoAtCallLevel(msg, 2, additionalFields...)
}

// LogWarning is a helper function that logs a warning
func LogInfoAtCallLevel(msg string, lvl int, additionalFields ...logrus.Fields) {
	_, file, line, _ := runtime.Caller(lvl) // Get the caller of this function

	// Create base fields
	logFields := logrus.Fields{
		"file": filepath.Join(filepath.Base(filepath.Dir(file)), filepath.Base(file)),
		"line": line,
	}

	// Merge any additional fields
	if len(additionalFields) > 0 {
		maps.Copy(logFields, additionalFields[0])
	}

	logrus.WithFields(logFields).Info(msg)
}

// LogDebug is a helper function that logs a debug message
func LogDebug(msg string, additionalFields ...logrus.Fields) {
	LogDebugAtCallLevel(msg, 2, additionalFields...)
}

// LogDebugAtCallLevel is a helper function that logs a debug message at a specific call level
func LogDebugAtCallLevel(msg string, lvl int, additionalFields ...logrus.Fields) {
	_, file, line, _ := runtime.Caller(lvl) // Get the caller of this function

	// Create base fields
	logFields := logrus.Fields{
		"file": filepath.Join(filepath.Base(filepath.Dir(file)), filepath.Base(file)),
		"line": line,
	}

	// Merge any additional fields
	if len(additionalFields) > 0 {
		maps.Copy(logFields, additionalFields[0])
	}

	logrus.WithFields(logFields).Debug(msg)
}
