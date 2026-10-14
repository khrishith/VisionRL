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
    // Generates Assets/VisionRL/Scenes/EscapeRoom.unity. Never touches GridWorld.unity.
    // World space: cell (x,z) centre = (x-3.5, z-3.5). Interior spans -4..4; walls are 0.5 thick outside it.
    public static class EscapeRoomBuilder
    {
        public const string ScenePath = "Assets/VisionRL/Scenes/EscapeRoom.unity";
        const string RobotPrefab = "Assets/VisionRL/Prefabs/RobotVisual.prefab";
        const float WallHeight = 1f;
        // Stage 6: floor checkerboard flattened, obstacle tops much lighter, so obstacles read at 84x84.
        static readonly Color FloorAColor = new Color(.15f, .16f, .20f);
        static readonly Color FloorBColor = new Color(.175f, .185f, .23f);
        static readonly Color ObstacleColor = new Color(.52f, .58f, .68f);

        // Re-applies the three colours to the existing material assets (the scene keeps using them).
        [MenuItem("VisionRL/Escape Room/Refresh Materials")]
        public static void RefreshMaterials()
        {
            Lit("RoomFloorA", FloorAColor);
            Lit("RoomFloorB", FloorBColor);
            Lit("RoomObstacle", ObstacleColor);
            AssetDatabase.SaveAssets();
            Debug.Log("VISIONRL_MATERIALS_REFRESHED");
        }

        [MenuItem("VisionRL/Escape Room/Create or Open Scene")]
        public static void CreateScene()
        {
            if (File.Exists(ScenePath)) { EditorSceneManager.OpenScene(ScenePath); return; }
            if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            Directory.CreateDirectory("Assets/VisionRL/Materials");

            var lit = Shader.Find("Universal Render Pipeline/Lit");
            var floorA = Lit("RoomFloorA", FloorAColor);
            var floorB = Lit("RoomFloorB", FloorBColor);
            var slab = Lit("RoomSlab", new Color(.06f, .06f, .08f));
            var wall = Lit("RoomWall", new Color(.12f, .13f, .17f));
            var obstacle = Lit("RoomObstacle", ObstacleColor);
            var green = AssetDatabase.LoadAssetAtPath<Material>("Assets/VisionRL/Materials/TargetGreen.mat");
            var cyan = AssetDatabase.LoadAssetAtPath<Material>("Assets/VisionRL/Materials/RobotCyan.mat");
            if (green == null || cyan == null || lit == null) throw new Exception("Required materials/shader missing");

            // Lighting: soft flat ambient plus one shadow-casting sun, so walls and obstacles read in 3D.
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(.42f, .45f, .52f);
            var sun = new GameObject("Sun").AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.transform.rotation = Quaternion.Euler(55, -35, 0);
            sun.color = new Color(1f, .97f, .92f);
            sun.intensity = 1.1f;
            sun.shadows = LightShadows.Soft;
            sun.shadowStrength = .75f;

            var room = new GameObject("EscapeRoom");
            var floor = Child(room, "Floor");
            Box(floor, "Slab", new Vector3(0, -.2f, 0), new Vector3(8.2f, .2f, 8.2f), slab);
            for (int z = 0; z < RoomWorld.Size; z++)
            for (int x = 0; x < RoomWorld.Size; x++)
                Box(floor, $"Tile_{x}_{z}", Cell(x, z, -.08f), new Vector3(.96f, .1f, .96f), (x + z) % 2 == 0 ? floorA : floorB);

            // Walls. The north wall is split to leave a 2-cell exit above the goal cells (x=5,6 -> world x 1..3).
            var walls = Child(room, "Walls");
            float y = WallHeight / 2;
            Box(walls, "Wall_South", new Vector3(0, y, -4.25f), new Vector3(9f, WallHeight, .5f), wall);
            Box(walls, "Wall_West", new Vector3(-4.25f, y, 0), new Vector3(.5f, WallHeight, 8f), wall);
            Box(walls, "Wall_East", new Vector3(4.25f, y, 0), new Vector3(.5f, WallHeight, 8f), wall);
            Box(walls, "Wall_North_Left", new Vector3(-1.75f, y, 4.25f), new Vector3(5.5f, WallHeight, .5f), wall);
            Box(walls, "Wall_North_Right", new Vector3(3.75f, y, 4.25f), new Vector3(1.5f, WallHeight, .5f), wall);
            // Restrained cyan accent along the inner base of the walls.
            Box(walls, "Trim_South", new Vector3(0, .03f, -3.98f), new Vector3(8f, .04f, .03f), cyan);
            Box(walls, "Trim_West", new Vector3(-3.98f, .03f, 0), new Vector3(.03f, .04f, 8f), cyan);
            Box(walls, "Trim_East", new Vector3(3.98f, .03f, 0), new Vector3(.03f, .04f, 8f), cyan);
            Box(walls, "Trim_North_Left", new Vector3(-1.5f, .03f, 3.98f), new Vector3(5f, .04f, .03f), cyan);
            Box(walls, "Trim_North_Right", new Vector3(3.5f, .03f, 3.98f), new Vector3(1f, .04f, .03f), cyan);

            // Static obstacles come from the same layout the mechanics use.
            var obstacles = Child(room, "Obstacles");
            var world = new RoomWorld();
            for (int z = 0; z < RoomWorld.Size; z++)
            for (int x = 0; x < RoomWorld.Size; x++)
                if (world.IsObstacle(x, z))
                    Box(obstacles, $"Obstacle_{x}_{z}", Cell(x, z, .35f), new Vector3(.88f, .7f, .88f), obstacle);

            // Goal zone (the two goal cells) and exit frame in the wall gap.
            var goal = Child(room, "Goal");
            Box(goal, "GoalPad", new Vector3(2f, .0f, 3.5f), new Vector3(1.9f, .04f, .94f), green);
            Box(goal, "ExitFloor", new Vector3(2f, .0f, 4.25f), new Vector3(1.9f, .04f, .5f), green);
            Box(goal, "ExitLintel", new Vector3(2f, WallHeight + .06f, 4.25f), new Vector3(2.1f, .12f, .5f), green);
            Box(goal, "ExitPostLeft", new Vector3(.95f, y, 4.25f), new Vector3(.1f, WallHeight, .5f), green);
            Box(goal, "ExitPostRight", new Vector3(3.05f, y, 4.25f), new Vector3(.1f, WallHeight, .5f), green);

            // Observation camera: fixed, top-down, orthographic, shows the whole room and exit.
            var cameraObject = new GameObject("Observation Camera");
            var camera = cameraObject.AddComponent<Camera>();
            camera.transform.position = new Vector3(0, 10, 0);
            camera.transform.rotation = Quaternion.Euler(90, 0, 0);
            camera.orthographic = true;
            camera.orthographicSize = 4.6f;
            camera.aspect = 1f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(.03f, .03f, .04f);
            camera.nearClipPlane = .1f;
            camera.farClipPlane = 20f;
            camera.allowHDR = false;
            camera.allowMSAA = false;
            cameraObject.tag = "MainCamera";

            // Agent: same components as the GridWorld agent, with the robot prefab as the visual.
            var agentObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            agentObject.name = "Blue Agent";
            agentObject.transform.position = Cell(0, 0, .375f);
            agentObject.transform.localScale = Vector3.one * .75f;
            agentObject.SetActive(false); // configure sensors before the Agent enables
            var behavior = agentObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "VisionRL";
            behavior.BrainParameters.VectorObservationSize = 0;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(4);
            behavior.ObservableAttributeHandling = ObservableAttributeOptions.Ignore;
            behavior.UseChildSensors = false;
            var sensor = agentObject.AddComponent<CameraSensorComponent>();
            sensor.Camera = camera;
            sensor.SensorName = "RGB84";
            sensor.Width = 84;
            sensor.Height = 84;
            sensor.Grayscale = false;
            sensor.ObservationStacks = 1;
            sensor.CompressionType = SensorCompressionType.PNG;
            sensor.RuntimeCameraEnable = true;
            agentObject.AddComponent<RoomAgent>();
            var decision = agentObject.AddComponent<GridDecisionRequester>();
            decision.DecisionPeriod = 1;
            decision.TakeActionsBetweenDecisions = false;
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(RobotPrefab);
            if (prefab == null) throw new Exception("RobotVisual.prefab missing");
            var visual = (GameObject)PrefabUtility.InstantiatePrefab(prefab, agentObject.transform);
            visual.name = "Robot Visual";
            visual.transform.localPosition = Vector3.zero;
            visual.transform.localRotation = Quaternion.identity;
            visual.transform.localScale = Vector3.one;
            agentObject.GetComponent<MeshRenderer>().enabled = false; // cube hidden; its BoxCollider is the only collider
            agentObject.SetActive(true);

            Directory.CreateDirectory("Assets/VisionRL/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            Debug.Log("VISIONRL_ESCAPEROOM_CREATED " + ScenePath);
        }

        [MenuItem("VisionRL/Escape Room/Render Preview Images")]
        public static void RenderPreviews()
        {
            CreateScene();
            var go = new GameObject("PreviewCam");
            var cam = go.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(.03f, .03f, .04f);
            cam.fieldOfView = 40;
            Directory.CreateDirectory("Reports");
            var agent = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            agent.transform.position = Cell(1, 0, .375f);
            Shot(cam, new Vector3(-1f, 7.5f, -7.5f), new Vector3(0, 0, 0), "Reports/stage4-room-overview.png");
            Shot(cam, new Vector3(5.5f, 4.2f, -2.2f), new Vector3(0, .3f, 1.0f), "Reports/stage4-room-side.png");
            UnityEngine.Object.DestroyImmediate(go);
            agent.transform.position = Cell(0, 0, .375f); // restore; scene is not saved after previews
        }

        static void Shot(Camera cam, Vector3 pos, Vector3 look, string path)
        {
            cam.transform.position = pos;
            cam.transform.LookAt(look);
            var rt = new RenderTexture(768, 768, 24);
            cam.targetTexture = rt;
            cam.Render();
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            var tex = new Texture2D(768, 768, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, 768, 768), 0, 0);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            RenderTexture.active = prev;
            cam.targetTexture = null;
            UnityEngine.Object.DestroyImmediate(rt);
            UnityEngine.Object.DestroyImmediate(tex);
        }

        // Batch entry: create the scene and render previews.
        public static void BuildSceneAndPreviews()
        {
            RenderPreviews();
        }

        [MenuItem("VisionRL/Escape Room/Build macOS Player")]
        public static void BuildPlayer()
        {
            CreateScene();
            Directory.CreateDirectory("Builds");
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath }, locationPathName = "Builds/EscapeRoom.app",
                target = BuildTarget.StandaloneOSX, options = BuildOptions.None
            });
            if (report.summary.result != BuildResult.Succeeded) throw new Exception("EscapeRoom build failed: " + report.summary.result);
            Debug.Log("VISIONRL_ESCAPEROOM_BUILD_PASSED");
        }

        static Vector3 Cell(int x, int z, float y) => new Vector3(x - RoomAgent.CellOffset, y, z - RoomAgent.CellOffset);

        static GameObject Child(GameObject parent, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);
            return go;
        }

        static GameObject Box(GameObject parent, string name, Vector3 pos, Vector3 scale, Material m)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent.transform, false);
            go.transform.position = pos;
            go.transform.localScale = scale;
            go.GetComponent<Renderer>().sharedMaterial = m;
            return go;
        }

        static Material Lit(string name, Color color)
        {
            string path = $"Assets/VisionRL/Materials/{name}.mat";
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m == null)
            {
                m = new Material(Shader.Find("Universal Render Pipeline/Lit")) { name = name };
                AssetDatabase.CreateAsset(m, path);
            }
            m.SetColor("_BaseColor", color);
            m.SetFloat("_Smoothness", .25f);
            m.SetFloat("_Metallic", 0f);
            EditorUtility.SetDirty(m);
            return m;
        }
    }
}
