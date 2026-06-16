from hawkesnest.suites import EntanglementSuite, HeterogeneitySuite
from hawkesnest.config import SimulatorConfig

# Generate standard benchmark datasets
ent_data = EntanglementSuite().generate(level="L2", n_events=100, seed=42, out_dir="./outputs/data/ent_L2")
het_data = HeterogeneitySuite().generate(level="H3", n_events=100, seed=42, out_dir="./outputs/data/het_H3")

print("# Genearate standard benchmark")
print(ent_data.events.head())
print(het_data.events.head())

# Generate data using custom simulator
## 1. define customized simulation configuration 
sim_config = {
    "domain": {
        "type": "rectangle",
        "x_min": 0,
        "y_min": 0,
        "x_max": 1,
        "y_max": 1,
    },
    "backgrounds":[
        {
            "type":"constant",
            "rate":0.1
        }
    ],
    "kernels":[
        {
            "type":"separable",
            "temporal_decay":0.4,
            "spatial_sigma":0.05,
            "adjacency":[[0.2],],
            "lambda_max": 25.0,
        }
    ]
}

## 2. build simulator
simulator = SimulatorConfig.model_validate(sim_config).build()

## 3. generate data (simulate the events)
events, parents = simulator.simulate(n=100, seed=7, tau_max = 5)

print("# Generate customized data")
print(events.head())


#from hawkesnest.viz import plot_intensity_snapshots
#
#fig, axes = plot_intensity_snapshots(
    #lambda s, t: simulator.background(s, t, 1),
    #times=[0.0, 0.5, 1.0],
    #cmap="magma",
    #title="Background intensity mu(s,t)",
#)
#fig.savefig("outputs/intensity_snapshots.png", dpi=160)

from hawkesnest.viz import plot_events_2d

fig, ax = plot_events_2d(events, color_by="t", title="Events coloured by time")
fig.savefig("outputs/events_2d.png", dpi=160)