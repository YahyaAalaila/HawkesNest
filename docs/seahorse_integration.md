# Seahorse Integration

Seahorse uses synthetic corpora generated from HawkesNest. HawkesNest should be cited when Seahorse experiments depend on these synthetic suite3 or suite4 corpora.

The validated recipes preserve the working paper-generation path:

```text
EntanglementSuite / HeterogeneitySuite
-> hawkesnest.config.SimulatorConfig
-> SimulatorConfig.build()
-> hawkesnest.simulator.hawkes.HawkesSimulator
-> hawkesnest.utils.thinning.thinning
```

Recipe mode is the recommended path for Seahorse reproduction and paper-aligned stress tests:

```python
from hawkesnest.suites import EntanglementSuite, HeterogeneitySuite

ent = EntanglementSuite().generate(level="L2", n_events=50, seed=123)
het = HeterogeneitySuite().generate(level="H3", n_events=50, seed=123)

print(ent.events.head())
print(het.metadata["simulator_class"])
```

For schema details, see [output_schema.md](output_schema.md).
