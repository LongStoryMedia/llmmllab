import { SxProps, Theme } from "@mui/material";
interface IconProps {
    size?: number;
    sx?: SxProps<Theme> | undefined;
    withBeaker?: boolean;
    speed?: number;
}
declare const Icon: React.FC<IconProps>;
export default Icon;
