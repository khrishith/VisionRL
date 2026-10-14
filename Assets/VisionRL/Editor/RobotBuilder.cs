using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace VisionRL.Editor
{
    // Builds the low-poly robot prefab from primitives and attaches it to the Blue Agent.
    // The prefab is visual only: no colliders, no scripts. All RL logic stays on the agent root.
    public static class RobotBuilder
    {
        const string PrefabPath = "Assets/VisionRL/Prefabs/RobotVisual.prefab";
        const string ScenePath = "Assets/VisionRL/Scenes/GridWorld.unity";
        const string VisualName = "Robot Visual";

        // Local space of the agent root (scale 0.75): x/z in [-0.5, 0.5], y floor = -0.5. Front = +Z.
        [MenuItem("VisionRL/Robot/Create Robot Prefab")]
        public static void CreatePrefab()
        {
            Directory.CreateDirectory("Assets/VisionRL/Prefabs");
            var blue = Mat("RobotBlue", new Color(.04f, .25f, 1f));
            var white = Mat("RobotWhite", new Color(.93f, .95f, 1f));
            var dark = Mat("RobotDark", new Color(.07f, .08f, .11f));
            var cyan = Mat("RobotCyan", new Color(.1f, .95f, 1f));
            var tire = Mat("RobotTire", new Color(.14f, .15f, .18f));

            var root = new GameObject("RobotVisual");
            // Lower body
            Part(root, "Chassis", PrimitiveType.Cube, new Vector3(0, -.27f, 0), new Vector3(.76f, .22f, .92f), blue);
            Part(root, "FenderL", PrimitiveType.Cube, new Vector3(-.3f, -.14f, -.04f), new Vector3(.14f, .05f, .8f), white);
            Part(root, "FenderR", PrimitiveType.Cube, new Vector3(.3f, -.14f, -.04f), new Vector3(.14f, .05f, .8f), white);
            Part(root, "BumperFront", PrimitiveType.Cube, new Vector3(0, -.27f, .47f), new Vector3(.5f, .1f, .05f), white);
            Part(root, "BumperLight", PrimitiveType.Cube, new Vector3(0, -.27f, .5f), new Vector3(.3f, .03f, .02f), cyan);
            Part(root, "BumperRear", PrimitiveType.Cube, new Vector3(0, -.25f, -.47f), new Vector3(.5f, .1f, .05f), dark);
            // Wheels: cylinder axis along X
            foreach (var sx in new[] { -1f, 1f })
            foreach (var sz in new[] { -1f, 1f })
            {
                var n = (sx < 0 ? "L" : "R") + (sz < 0 ? "Rear" : "Front");
                Part(root, "Wheel" + n, PrimitiveType.Cylinder, new Vector3(sx * .43f, -.35f, sz * .3f),
                    new Vector3(.3f, .06f, .3f), tire, new Vector3(0, 0, 90));
                Part(root, "Hub" + n, PrimitiveType.Cylinder, new Vector3(sx * .44f, -.35f, sz * .3f),
                    new Vector3(.14f, .065f, .14f), white, new Vector3(0, 0, 90));
            }
            // Torso
            Part(root, "Torso", PrimitiveType.Cube, new Vector3(0, 0, -.05f), new Vector3(.5f, .18f, .5f), white);
            Part(root, "TorsoStripeL", PrimitiveType.Cube, new Vector3(-.255f, 0, -.05f), new Vector3(.012f, .05f, .34f), cyan);
            Part(root, "TorsoStripeR", PrimitiveType.Cube, new Vector3(.255f, 0, -.05f), new Vector3(.012f, .05f, .34f), cyan);
            // Head / sensor section
            Part(root, "Head", PrimitiveType.Cube, new Vector3(0, .2f, .02f), new Vector3(.56f, .24f, .5f), blue);
            Part(root, "HeadStripe", PrimitiveType.Cube, new Vector3(0, .322f, -.1f), new Vector3(.36f, .01f, .1f), white);
            Part(root, "FrontLightBar", PrimitiveType.Cube, new Vector3(0, .322f, .2f), new Vector3(.34f, .01f, .05f), cyan);
            Part(root, "EarL", PrimitiveType.Cube, new Vector3(-.31f, .2f, .0f), new Vector3(.06f, .14f, .2f), white);
            Part(root, "EarR", PrimitiveType.Cube, new Vector3(.31f, .2f, .0f), new Vector3(.06f, .14f, .2f), white);
            Part(root, "Visor", PrimitiveType.Cube, new Vector3(0, .2f, .275f), new Vector3(.46f, .15f, .04f), dark);
            Part(root, "LensRing", PrimitiveType.Cylinder, new Vector3(0, .2f, .3f), new Vector3(.19f, .02f, .19f), white, new Vector3(90, 0, 0));
            Part(root, "Lens", PrimitiveType.Cylinder, new Vector3(0, .2f, .32f), new Vector3(.13f, .02f, .13f), cyan, new Vector3(90, 0, 0));
            Part(root, "SensorL", PrimitiveType.Cube, new Vector3(-.17f, .2f, .298f), new Vector3(.05f, .05f, .02f), cyan);
            Part(root, "SensorR", PrimitiveType.Cube, new Vector3(.17f, .2f, .298f), new Vector3(.05f, .05f, .02f), cyan);
            // Antenna
            Part(root, "AntennaMast", PrimitiveType.Cylinder, new Vector3(.17f, .4f, -.14f), new Vector3(.03f, .08f, .03f), dark);
            Part(root, "AntennaTip", PrimitiveType.Sphere, new Vector3(.17f, .49f, -.14f), Vector3.one * .07f, cyan);

            PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
            Object.DestroyImmediate(root);
            AssetDatabase.SaveAssets();
            Debug.Log("VISIONRL_ROBOT_PREFAB " + PrefabPath);
        }

        [MenuItem("VisionRL/Robot/Apply Robot To GridWorld")]
        public static void ApplyToGridWorld()
        {
            if (AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath) == null) CreatePrefab();
            var scene = EditorSceneManager.GetActiveScene();
            if (scene.path != ScenePath) scene = EditorSceneManager.OpenScene(ScenePath);
            var agent = Object.FindAnyObjectByType<GridAgent>();
            if (agent == null) throw new System.Exception("GridAgent not found in GridWorld");
            if (agent.transform.Find(VisualName) == null)
            {
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
                var visual = (GameObject)PrefabUtility.InstantiatePrefab(prefab, agent.transform);
                visual.name = VisualName;
                visual.transform.localPosition = Vector3.zero;
                visual.transform.localRotation = Quaternion.identity;
                visual.transform.localScale = Vector3.one;
            }
            // Original cube stays on the agent as a backup: only its mesh is hidden.
            // Its BoxCollider remains the single collider. Re-enable the MeshRenderer to revert.
            agent.GetComponent<MeshRenderer>().enabled = false;
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
            Debug.Log("VISIONRL_ROBOT_APPLIED");
        }

        // Batch entry point: prefab, scene, preview images.
        public static void BuildAll()
        {
            CreatePrefab();
            ApplyToGridWorld();
            RenderPreviews();
        }

        [MenuItem("VisionRL/Robot/Render Preview Images")]
        public static void RenderPreviews()
        {
            var agent = Object.FindAnyObjectByType<GridAgent>();
            var go = new GameObject("PreviewCam");
            var cam = go.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(.09f, .09f, .11f);
            cam.fieldOfView = 35;
            Directory.CreateDirectory("Reports");
            var p = agent.transform.position;
            Shot(cam, p + new Vector3(1.3f, 1.0f, 2.2f), p + Vector3.up * .1f, "Reports/stage3-robot-front.png");
            Shot(cam, p + new Vector3(-1.6f, 1.2f, -1.8f), p + Vector3.up * .1f, "Reports/stage3-robot-rear.png");
            Object.DestroyImmediate(go);
        }

        static void Shot(Camera cam, Vector3 pos, Vector3 look, string path)
        {
            cam.transform.position = pos;
            cam.transform.LookAt(look);
            var rt = new RenderTexture(512, 512, 24);
            cam.targetTexture = rt;
            cam.Render();
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            var tex = new Texture2D(512, 512, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, 512, 512), 0, 0);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            RenderTexture.active = prev;
            cam.targetTexture = null;
            Object.DestroyImmediate(rt);
            Object.DestroyImmediate(tex);
        }

        static void Part(GameObject parent, string name, PrimitiveType type, Vector3 pos, Vector3 scale, Material m, Vector3 euler = default)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            Object.DestroyImmediate(go.GetComponent<Collider>()); // visuals must never collide
            go.transform.SetParent(parent.transform, false);
            go.transform.localPosition = pos;
            go.transform.localEulerAngles = euler;
            go.transform.localScale = scale;
            go.GetComponent<Renderer>().sharedMaterial = m;
            go.GetComponent<Renderer>().shadowCastingMode = ShadowCastingMode.Off;
        }

        // Same flat URP Unlit look as the floor/target materials (the scene has no lights).
        static Material Mat(string name, Color color)
        {
            string path = $"Assets/VisionRL/Materials/{name}.mat";
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m == null)
            {
                m = new Material(Shader.Find("Universal Render Pipeline/Unlit")) { name = name };
                AssetDatabase.CreateAsset(m, path);
            }
            m.SetColor("_BaseColor", color);
            EditorUtility.SetDirty(m);
            return m;
        }
    }
}
