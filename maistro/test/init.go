package test

import (
	"context"
	"fmt"
	"log"
	"maistro/config"
	"maistro/storage"
	"maistro/util"
	"testing"

	"github.com/sirupsen/logrus"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
)

type TestFunc func(t *testing.T)

type SetupFunc func() error

func LLmmLLab_Test(t *testing.T, s SetupFunc, tests ...TestFunc) {
	confFile := "testdata/.config.yaml"
	ctx := context.Background()

	pgc, err := postgres.Run(ctx,
		"timescale/timescaledb-ha:pg17",
		postgres.WithInitScripts("testdata/init_test_db.sh"),
		postgres.BasicWaitStrategies(),
	)
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := testcontainers.TerminateContainer(pgc); err != nil {
			log.Printf("failed to terminate container: %s", err)
		}
	}()

	config.GetConfig(&confFile)

	psqlconn, err := pgc.ConnectionString(ctx)
	if err != nil {
		panic("failed to get connection string")
	}

	storage.InitializeStorage()

	if err := storage.InitDB(psqlconn); err != nil {
		util.HandleError(err)
	}
	util.LogInfo("Connected to PostgreSQL database", logrus.Fields{
		"connection": psqlconn,
	})

	if s != nil {
		if err := s(); err != nil {
			t.Fatalf("Setup function failed: %v", err)
		}
	}

	for _, test := range tests {
		t.Run("LLmmLLab", func(tt *testing.T) {
			test(tt)
		})
	}
}

func Init() {
	confFile := "testdata/.config.yaml"
	conf := config.GetConfig(&confFile)

	psqlconn := fmt.Sprintf(
		"postgresql://%s:%s@%s:%d/%s?sslmode=%s",
		conf.Database.User,
		conf.Database.Password,
		conf.Database.Host,
		conf.Database.Port,
		conf.Database.DBName,
		conf.Database.SSLMode,
	)

	InitMockStore()

	if err := storage.InitDB(psqlconn); err != nil {
		util.HandleError(err)
	}
	util.LogInfo("Connected to PostgreSQL database", logrus.Fields{
		"connection": psqlconn,
	})
}
