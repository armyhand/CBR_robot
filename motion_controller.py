
class controller():
    def PD_control(last_pos, cur_pos, goal_pos, goal_vel, kp, kd): #PD控制生成控制指令
        cur_vel = (cur_pos - last_pos) * 20
        a = kp * (goal_pos - cur_pos) + kd * (goal_vel - cur_vel)
        return a

    def vel_control(cur_pos, goal_pos, k):
        a = k*(goal_pos - cur_pos)

        return a

    def Force_control(xe, dxe, Fe, Mp, Bp, Cp, freq):
        ddxe = (Fe - Bp * dxe - Cp * xe) / Mp
        dxe_next = dxe + ddxe / freq
        xe_next = xe + dxe_next / freq

        return xe_next