package util

import (
	"path/filepath"
	"runtime"

	"maps"

	"github.com/sirupsen/logrus"
)

// HandleFatalError is a helper function that logs the error and panics
func HandleFatalError(err error, additionalFields ...logrus.Fields) {
	if err == nil {
		return
	}
	_, file, line, _ := runtime.Caller(1) // Get the caller of this function

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
	if err == nil {
		return nil
	}
	_, file, line, _ := runtime.Caller(1) // Get the caller of this function

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
	_, file, line, _ := runtime.Caller(1) // Get the caller of this function

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
	_, file, line, _ := runtime.Caller(1) // Get the caller of this function

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
	_, file, line, _ := runtime.Caller(1) // Get the caller of this function

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
