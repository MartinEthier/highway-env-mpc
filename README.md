# highway-env-mpc
Application of model predictive control (MPC) on the highway-env simulator. Controller takes into account predicted trajectories for all other agents in the scene.

## Register Env
```
pip install -e .
```

## 
Convert image folder into video:
```
ffmpeg -r 10 -i img_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
```
