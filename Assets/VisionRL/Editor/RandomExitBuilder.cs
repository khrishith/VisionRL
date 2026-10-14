using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace VisionRL.Editor
{
    // Stage 8: generates Assets/VisionRL/Scenes/EscapeRoomRandomExit.unity. The Stage 4-7 EscapeRoom.unity
    // (fixed exit, trained model assigned) is never touched. Same room, lights, camera and robot; walls are
    // solid and the exit (green pad + door slab) is moved to a random border cell by RoomAgent at every reset.
    public static class RandomExitBuilder
    {
        public const string ScenePath = "Assets/VisionRL/Scenes/EscapeRoomRandomExit.unity";
        public const int Horizon = 48; // blind policies reach <=~17%; farthest optimal route is 14 steps
        const string RobotPrefab = "Assets/VisionRL/Prefabs/RobotVisual.prefab";
        const float WallHeight = 1f;

        [MenuItem("VisionRL/Random Exit/Create or Open Scene")]
        public static void CreateScene()
        {
            if (File.Exists(ScenePath)) { EditorSceneManager.OpenScene(ScenePath); return; }
            if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            Material M(string n) { var m = AssetDatabase.LoadAssetAtPath<Material>($"Assets/VisionRL/Materials/{n}.mat"); if (m == null) throw new Exception("missing material " + n); return m; }
            var floorA = M("RoomFloorA"); var floorB = M("RoomFloorB"); var slab = M("RoomSlab"); var wall = M("RoomWall");
            var obstacle = M("RoomObstacle"); var green = M("TargetGreen"); var cyan = M("RobotCyan");

            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(.42f, .45f, .52f);
            var sun = new GameObject("Sun").AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.transform.rotation = Quaternion.Euler(55, -35, 0);
            sun.color = new Color(1f, .97f, .92f);
            sun.intensity = 1.1f; sun.shadows = LightShadows.Soft; sun.shadowStrength = .75f;

            var room = new GameObject("EscapeRoom");
            var floor = Child(room, "Floor");
            Box(floor, "Slab", new Vector3(0, -.2f, 0), new Vector3(8.2f, .2f, 8.2f), slab);
            for (int z = 0; z < RoomWorld.Size; z++)
            for (int x = 0; x < RoomWorld.Size; x++)
                Box(floor, $"Tile_{x}_{z}", Cell(x, z, -.08f), new Vector3(.96f, .1f, .96f), (x + z) % 2 == 0 ? floorA : floorB);

            // Four SOLID walls: the only opening-like feature is the green door the agent moves to the exit.
            var walls = Child(room, "Walls"); float y = WallHeight / 2;
            Box(walls, "Wall_South", new Vector3(0, y, -4.25f), new Vector3(9f, WallHeight, .5f), wall);
            Box(walls, "Wall_North", new Vector3(0, y, 4.25f), new Vector3(9f, WallHeight, .5f), wall);
            Box(walls, "Wall_West", new Vector3(-4.25f, y, 0), new Vector3(.5f, WallHeight, 8f), wall);
            Box(walls, "Wall_East", new Vector3(4.25f, y, 0), new Vector3(.5f, WallHeight, 8f), wall);
            Box(walls, "Trim_South", new Vector3(0, .03f, -3.98f), new Vector3(8f, .04f, .03f), cyan);
            Box(walls, "Trim_North", new Vector3(0, .03f, 3.98f), new Vector3(8f, .04f, .03f), cyan);
            Box(walls, "Trim_West", new Vector3(-3.98f, .03f, 0), new Vector3(.03f, .04f, 8f), cyan);
            Box(walls, "Trim_East", new Vector3(3.98f, .03f, 0), new Vector3(.03f, .04f, 8f), cyan);

            var obstacles = Child(room, "Obstacles");
            var layout = new RoomWorld(RoomWorld.RandomExitLayout, true);
            for (int z = 0; z < RoomWorld.Size; z++)
            for (int x = 0; x < RoomWorld.Size; x++)
                if (layout.IsObstacle(x, z)) Box(obstacles, $"Obstacle_{x}_{z}", Cell(x, z, .35f), new Vector3(.88f, .7f, .88f), obstacle);

            // Exit marker: positions below are placeholders; RoomAgent.SyncExitVisual moves them at every reset.
            var goal = Child(room, "Goal");
            var pad = Box(goal, "GoalPad", new Vector3(2f, 0f, 3.5f), new Vector3(.94f, .04f, .94f), green);
            var door = Box(goal, "ExitDoor", new Vector3(2f, y, 4.25f), new Vector3(.9f, WallHeight + .08f, .52f), green);

            var cameraObject = new GameObject("Observation Camera");
            var camera = cameraObject.AddComponent<Camera>();
            camera.transform.position = new Vector3(0, 10, 0);
            camera.transform.rotation = Quaternion.Euler(90, 0, 0);
            camera.orthographic = true; camera.orthographicSize = 4.6f; camera.aspect = 1f;
            camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = new Color(.03f, .03f, .04f);
            camera.nearClipPlane = .1f; camera.farClipPlane = 20f; camera.allowHDR = false; camera.allowMSAA = false;
            cameraObject.tag = "MainCamera";

            var agentObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            agentObject.name = "Blue Agent";
            agentObject.transform.position = Cell(0, 0, .375f);
            agentObject.transform.localScale = Vector3.one * .75f;
            agentObject.SetActive(false);
            var behavior = agentObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "VisionRL";
            behavior.BrainParameters.VectorObservationSize = 0;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(4);
            behavior.ObservableAttributeHandling = ObservableAttributeOptions.Ignore;
            behavior.UseChildSensors = false;
            var sensor = agentObject.AddComponent<CameraSensorComponent>();
            sensor.Camera = camera; sensor.SensorName = "RGB84"; sensor.Width = 84; sensor.Height = 84;
            sensor.Grayscale = false; sensor.ObservationStacks = 1; sensor.CompressionType = SensorCompressionType.PNG; sensor.RuntimeCameraEnable = true;
            var agent = agentObject.AddComponent<RoomAgent>();
            agent.randomizeExit = true; agent.minStartDistance = 6; agent.horizon = Horizon;
            agent.goalPad = pad.transform; agent.exitDoor = door.transform;
            var decision = agentObject.AddComponent<GridDecisionRequester>();
            decision.DecisionPeriod = 1; decision.TakeActionsBetweenDecisions = false;
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(RobotPrefab);
            if (prefab == null) throw new Exception("RobotVisual.prefab missing");
            var visual = (GameObject)PrefabUtility.InstantiatePrefab(prefab, agentObject.transform);
            visual.name = "Robot Visual";
            visual.transform.localPosition = Vector3.zero; visual.transform.localRotation = Quaternion.identity; visual.transform.localScale = Vector3.one;
            agentObject.GetComponent<MeshRenderer>().enabled = false;
            agentObject.SetActive(true);

            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            Debug.Log("VISIONRL_RANDOMEXIT_CREATED " + ScenePath);
        }

        [MenuItem("VisionRL/Random Exit/Build macOS Player")]
        public static void BuildPlayer()
        {
            CreateScene();
            Directory.CreateDirectory("Builds");
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath }, locationPathName = "Builds/EscapeRoomRandomExit.app",
                target = BuildTarget.StandaloneOSX, options = BuildOptions.None
            });
            if (report.summary.result != BuildResult.Succeeded) throw new Exception("RandomExit build failed: " + report.summary.result);
            Debug.Log("VISIONRL_RANDOMEXIT_BUILD_PASSED");
        }

        static Vector3 Cell(int x, int z, float y) => new Vector3(x - RoomAgent.CellOffset, y, z - RoomAgent.CellOffset);
        static GameObject Child(GameObject parent, string name) { var go = new GameObject(name); go.transform.SetParent(parent.transform, false); return go; }
        static GameObject Box(GameObject parent, string name, Vector3 pos, Vector3 scale, Material m)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name; go.transform.SetParent(parent.transform, false);
            go.transform.position = pos; go.transform.localScale = scale;
            go.GetComponent<Renderer>().sharedMaterial = m;
            return go;
        }
    }
}
