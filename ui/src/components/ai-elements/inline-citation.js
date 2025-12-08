"use client";
import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { Badge } from "../ui/badge";
import { Carousel, CarouselContent, CarouselItem } from "../ui/carousel";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { cn } from "../../lib/utils";
import { ArrowLeftIcon, ArrowRightIcon } from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
export const InlineCitation = ({ className, ...props }) => (_jsx("span", { className: cn("group inline items-center gap-1", className), ...props }));
export const InlineCitationText = ({ className, ...props }) => (_jsx("span", { className: cn("transition-colors group-hover:bg-accent", className), ...props }));
export const InlineCitationCard = (props) => (_jsx(HoverCard, { closeDelay: 0, openDelay: 0, ...props }));
export const InlineCitationCardTrigger = ({ sources, className, ...props }) => (_jsx(HoverCardTrigger, { asChild: true, children: _jsx(Badge, { className: cn("ml-1 rounded-full", className), variant: "secondary", ...props, children: sources[0] ? (_jsxs(_Fragment, { children: [new URL(sources[0]).hostname, " ", sources.length > 1 && `+${sources.length - 1}`] })) : ("unknown") }) }));
export const InlineCitationCardBody = ({ className, ...props }) => (_jsx(HoverCardContent, { className: cn("relative w-80 p-0", className), ...props }));
const CarouselApiContext = createContext(undefined);
const useCarouselApi = () => {
    const context = useContext(CarouselApiContext);
    return context;
};
export const InlineCitationCarousel = ({ className, children, ...props }) => {
    const [api, setApi] = useState();
    return (_jsx(CarouselApiContext.Provider, { value: api, children: _jsx(Carousel, { className: cn("w-full", className), setApi: setApi, ...props, children: children }) }));
};
export const InlineCitationCarouselContent = (props) => _jsx(CarouselContent, { ...props });
export const InlineCitationCarouselItem = ({ className, ...props }) => (_jsx(CarouselItem, { className: cn("w-full space-y-2 p-4 pl-8", className), ...props }));
export const InlineCitationCarouselHeader = ({ className, ...props }) => (_jsx("div", { className: cn("flex items-center justify-between gap-2 rounded-t-md bg-secondary p-2", className), ...props }));
export const InlineCitationCarouselIndex = ({ children, className, ...props }) => {
    const api = useCarouselApi();
    const [current, setCurrent] = useState(0);
    const [count, setCount] = useState(0);
    useEffect(() => {
        if (!api) {
            return;
        }
        setCount(api.scrollSnapList().length);
        setCurrent(api.selectedScrollSnap() + 1);
        api.on("select", () => {
            setCurrent(api.selectedScrollSnap() + 1);
        });
    }, [api]);
    return (_jsx("div", { className: cn("flex flex-1 items-center justify-end px-3 py-1 text-muted-foreground text-xs", className), ...props, children: children ?? `${current}/${count}` }));
};
export const InlineCitationCarouselPrev = ({ className, ...props }) => {
    const api = useCarouselApi();
    const handleClick = useCallback(() => {
        if (api) {
            api.scrollPrev();
        }
    }, [api]);
    return (_jsx("button", { "aria-label": "Previous", className: cn("shrink-0", className), onClick: handleClick, type: "button", ...props, children: _jsx(ArrowLeftIcon, { className: "size-4 text-muted-foreground" }) }));
};
export const InlineCitationCarouselNext = ({ className, ...props }) => {
    const api = useCarouselApi();
    const handleClick = useCallback(() => {
        if (api) {
            api.scrollNext();
        }
    }, [api]);
    return (_jsx("button", { "aria-label": "Next", className: cn("shrink-0", className), onClick: handleClick, type: "button", ...props, children: _jsx(ArrowRightIcon, { className: "size-4 text-muted-foreground" }) }));
};
export const InlineCitationSource = ({ title, url, description, className, children, ...props }) => (_jsxs("div", { className: cn("space-y-1", className), ...props, children: [title && (_jsx("h4", { className: "truncate font-medium text-sm leading-tight", children: title })), url && (_jsx("p", { className: "truncate break-all text-muted-foreground text-xs", children: url })), description && (_jsx("p", { className: "line-clamp-3 text-muted-foreground text-sm leading-relaxed", children: description })), children] }));
export const InlineCitationQuote = ({ children, className, ...props }) => (_jsx("blockquote", { className: cn("border-muted border-l-2 pl-3 text-muted-foreground text-sm italic", className), ...props, children: children }));
