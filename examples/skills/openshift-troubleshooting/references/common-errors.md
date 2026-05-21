# Common OpenShift Errors

## ImagePullBackOff

**Symptoms**: Pod stuck in `ImagePullBackOff` or `ErrImagePull` status.

**Causes**:
- Image does not exist in the registry
- Authentication failure (missing or expired pull secret)
- Network connectivity issues to the registry

**Resolution**:
1. Verify the image name and tag: `oc get pod <pod> -o jsonpath='{.spec.containers[*].image}'`
2. Check pull secrets: `oc get secrets -n <namespace> | grep pull`
3. Test registry access: `oc debug node/<node> -- chroot /host podman pull <image>`

## CrashLoopBackOff

**Symptoms**: Pod repeatedly starts and crashes.

**Causes**:
- Application error on startup
- Missing configuration (environment variables, config maps)
- Insufficient memory causing OOM kills

**Resolution**:
1. Check logs: `oc logs <pod> --previous`
2. Check for OOM: `oc get pod <pod> -o jsonpath='{.status.containerStatuses[*].lastState.terminated.reason}'`
3. Verify config: `oc get configmap -n <namespace>` and `oc get secret -n <namespace>`

## Pending Pods

**Symptoms**: Pod stays in `Pending` state.

**Causes**:
- Insufficient cluster resources (CPU, memory)
- Node selector or affinity rules that cannot be satisfied
- PersistentVolumeClaim not bound

**Resolution**:
1. Check events: `oc describe pod <pod> | grep -A5 Events`
2. Check node resources: `oc adm top nodes`
3. Check PVC status: `oc get pvc -n <namespace>`

## Service Unreachable

**Symptoms**: Cannot connect to a service from within or outside the cluster.

**Causes**:
- Service selector does not match pod labels
- Pod is not in `Running` state
- NetworkPolicy blocking traffic
- Route not configured (for external access)

**Resolution**:
1. Verify service selector: `oc get svc <service> -o jsonpath='{.spec.selector}'`
2. Check matching pods: `oc get pods -l <selector>`
3. Check network policies: `oc get networkpolicy -n <namespace>`
4. For external access, check routes: `oc get route -n <namespace>`
