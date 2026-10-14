# Local Unity 6.6 compatibility patch

Base: official Unity registry `com.unity.ml-agents` 4.0.0, resolved from
`com.unity.ml-agents@592fae96fab2` on 2026-10-13.

Unity 6000.6.5f1 compilation failed with CS0619 in
`Runtime/Integrations/Match3/Match3ActuatorComponent.cs` line 61.
`Object.GetInstanceID()` is an error-level obsolete API in this editor.

The only runtime-source change replaces `gameObject.GetInstanceID()` with
`gameObject.GetEntityId().GetHashCode()` to supply the helper's integer random seed.
VisionRL does not use Match3. PPO, sensors, communication, agent actions, and model
inference are unchanged. This embedded package and its existing LICENSE.md must
be retained with the project. Recheck this patch before upgrading Unity/packages.
