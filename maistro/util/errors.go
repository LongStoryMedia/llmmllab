package util

import (
	"fmt"

	"github.com/sirupsen/logrus"
)

// Error represents an application error
type Error struct {
	message string
	cause   error
	fields  logrus.Fields
}

// NewError creates a new Error
func NewError(message string) *Error {
	return &Error{
		message: message,
		fields:  logrus.Fields{},
	}
}

// WithCause adds a cause to the error
func (e *Error) WithCause(cause error) *Error {
	e.cause = cause
	return e
}

// WithField adds a field to the error
func (e *Error) WithField(key string, value interface{}) *Error {
	e.fields[key] = value
	return e
}

// Error returns the error message
func (e *Error) Error() string {
	if e.cause != nil {
		return fmt.Sprintf("%s: %v", e.message, e.cause)
	}
	return e.message
}

// Cause returns the underlying cause
func (e *Error) Cause() error {
	return e.cause
}

// Fields returns the error fields
func (e *Error) Fields() logrus.Fields {
	return e.fields
}
