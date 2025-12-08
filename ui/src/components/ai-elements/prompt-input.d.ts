import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList, CommandSeparator } from "../ui/command";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem } from "../ui/dropdown-menu";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { InputGroupAddon, InputGroupButton, InputGroupTextarea } from "../ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import type { ChatStatus, FileUIPart } from "ai";
import { type ComponentProps, type FormEvent, type HTMLAttributes, type PropsWithChildren, type ReactNode, type RefObject } from "react";
export type AttachmentsContext = {
    files: (FileUIPart & {
        id: string;
    })[];
    add: (files: File[] | FileList) => void;
    remove: (id: string) => void;
    clear: () => void;
    openFileDialog: () => void;
    fileInputRef: RefObject<HTMLInputElement | null>;
};
export type TextInputContext = {
    value: string;
    setInput: (v: string) => void;
    clear: () => void;
};
export type PromptInputControllerProps = {
    textInput: TextInputContext;
    attachments: AttachmentsContext;
    /** INTERNAL: Allows PromptInput to register its file textInput + "open" callback */
    __registerFileInput: (ref: RefObject<HTMLInputElement | null>, open: () => void) => void;
};
export declare const usePromptInputController: () => PromptInputControllerProps;
export declare const useProviderAttachments: () => AttachmentsContext;
export type PromptInputProviderProps = PropsWithChildren<{
    initialInput?: string;
}>;
/**
 * Optional global provider that lifts PromptInput state outside of PromptInput.
 * If you don't use it, PromptInput stays fully self-managed.
 */
export declare function PromptInputProvider({ initialInput: initialTextInput, children }: PromptInputProviderProps): import("react/jsx-runtime").JSX.Element;
export declare const usePromptInputAttachments: () => AttachmentsContext;
export type PromptInputAttachmentProps = HTMLAttributes<HTMLDivElement> & {
    data: FileUIPart & {
        id: string;
    };
    className?: string;
};
export declare function PromptInputAttachment({ data, className, ...props }: PromptInputAttachmentProps): import("react/jsx-runtime").JSX.Element;
export type PromptInputAttachmentsProps = Omit<HTMLAttributes<HTMLDivElement>, "children"> & {
    children: (attachment: FileUIPart & {
        id: string;
    }) => ReactNode;
};
export declare function PromptInputAttachments({ children, className, ...props }: PromptInputAttachmentsProps): import("react/jsx-runtime").JSX.Element | null;
export type PromptInputActionAddAttachmentsProps = ComponentProps<typeof DropdownMenuItem> & {
    label?: string;
};
export declare const PromptInputActionAddAttachments: ({ label, ...props }: PromptInputActionAddAttachmentsProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputMessage = {
    text: string;
    files: FileUIPart[];
};
export type PromptInputProps = Omit<HTMLAttributes<HTMLFormElement>, "onSubmit" | "onError"> & {
    accept?: string;
    multiple?: boolean;
    globalDrop?: boolean;
    syncHiddenInput?: boolean;
    maxFiles?: number;
    maxFileSize?: number;
    onError?: (err: {
        code: "max_files" | "max_file_size" | "accept";
        message: string;
    }) => void;
    onSubmit: (message: PromptInputMessage, event: FormEvent<HTMLFormElement>) => void | Promise<void>;
};
export declare const PromptInput: ({ className, accept, multiple, globalDrop, syncHiddenInput, maxFiles, maxFileSize, onError, onSubmit, children, ...props }: PromptInputProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputBodyProps = HTMLAttributes<HTMLDivElement>;
export declare const PromptInputBody: ({ className, ...props }: PromptInputBodyProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputTextareaProps = ComponentProps<typeof InputGroupTextarea>;
export declare const PromptInputTextarea: ({ onChange, className, placeholder, ...props }: PromptInputTextareaProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputHeaderProps = Omit<ComponentProps<typeof InputGroupAddon>, "align">;
export declare const PromptInputHeader: ({ className, ...props }: PromptInputHeaderProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputFooterProps = Omit<ComponentProps<typeof InputGroupAddon>, "align">;
export declare const PromptInputFooter: ({ className, ...props }: PromptInputFooterProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputToolsProps = HTMLAttributes<HTMLDivElement>;
export declare const PromptInputTools: ({ className, ...props }: PromptInputToolsProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputButtonProps = ComponentProps<typeof InputGroupButton>;
export declare const PromptInputButton: ({ variant, className, size, ...props }: PromptInputButtonProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputActionMenuProps = ComponentProps<typeof DropdownMenu>;
export declare const PromptInputActionMenu: (props: PromptInputActionMenuProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputActionMenuTriggerProps = PromptInputButtonProps;
export declare const PromptInputActionMenuTrigger: ({ className, children, ...props }: PromptInputActionMenuTriggerProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputActionMenuContentProps = ComponentProps<typeof DropdownMenuContent>;
export declare const PromptInputActionMenuContent: ({ className, ...props }: PromptInputActionMenuContentProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputActionMenuItemProps = ComponentProps<typeof DropdownMenuItem>;
export declare const PromptInputActionMenuItem: ({ className, ...props }: PromptInputActionMenuItemProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputSubmitProps = ComponentProps<typeof InputGroupButton> & {
    status?: ChatStatus;
};
export declare const PromptInputSubmit: ({ className, variant, size, status, children, ...props }: PromptInputSubmitProps) => import("react/jsx-runtime").JSX.Element;
interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start(): void;
    stop(): void;
    onstart: ((this: SpeechRecognition, ev: Event) => unknown) | null;
    onend: ((this: SpeechRecognition, ev: Event) => unknown) | null;
    onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => unknown) | null;
    onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => unknown) | null;
}
interface SpeechRecognitionEvent extends Event {
    results: SpeechRecognitionResultList;
    resultIndex: number;
}
type SpeechRecognitionResultList = {
    readonly length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
};
type SpeechRecognitionResult = {
    readonly length: number;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
    isFinal: boolean;
};
type SpeechRecognitionAlternative = {
    transcript: string;
    confidence: number;
};
interface SpeechRecognitionErrorEvent extends Event {
    error: string;
}
declare global {
    interface Window {
        SpeechRecognition: {
            new (): SpeechRecognition;
        };
        webkitSpeechRecognition: {
            new (): SpeechRecognition;
        };
    }
}
export type PromptInputSpeechButtonProps = ComponentProps<typeof PromptInputButton> & {
    textareaRef?: RefObject<HTMLTextAreaElement | null>;
    onTranscriptionChange?: (text: string) => void;
};
export declare const PromptInputSpeechButton: ({ className, textareaRef, onTranscriptionChange, ...props }: PromptInputSpeechButtonProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputSelectProps = ComponentProps<typeof Select>;
export declare const PromptInputSelect: (props: PromptInputSelectProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputSelectTriggerProps = ComponentProps<typeof SelectTrigger>;
export declare const PromptInputSelectTrigger: ({ className, ...props }: PromptInputSelectTriggerProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputSelectContentProps = ComponentProps<typeof SelectContent>;
export declare const PromptInputSelectContent: ({ className, ...props }: PromptInputSelectContentProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputSelectItemProps = ComponentProps<typeof SelectItem>;
export declare const PromptInputSelectItem: ({ className, ...props }: PromptInputSelectItemProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputSelectValueProps = ComponentProps<typeof SelectValue>;
export declare const PromptInputSelectValue: ({ className, ...props }: PromptInputSelectValueProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputHoverCardProps = ComponentProps<typeof HoverCard>;
export declare const PromptInputHoverCard: ({ openDelay, closeDelay, ...props }: PromptInputHoverCardProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputHoverCardTriggerProps = ComponentProps<typeof HoverCardTrigger>;
export declare const PromptInputHoverCardTrigger: (props: PromptInputHoverCardTriggerProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputHoverCardContentProps = ComponentProps<typeof HoverCardContent>;
export declare const PromptInputHoverCardContent: ({ align, ...props }: PromptInputHoverCardContentProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputTabsListProps = HTMLAttributes<HTMLDivElement>;
export declare const PromptInputTabsList: ({ className, ...props }: PromptInputTabsListProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputTabProps = HTMLAttributes<HTMLDivElement>;
export declare const PromptInputTab: ({ className, ...props }: PromptInputTabProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputTabLabelProps = HTMLAttributes<HTMLHeadingElement>;
export declare const PromptInputTabLabel: ({ className, ...props }: PromptInputTabLabelProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputTabBodyProps = HTMLAttributes<HTMLDivElement>;
export declare const PromptInputTabBody: ({ className, ...props }: PromptInputTabBodyProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputTabItemProps = HTMLAttributes<HTMLDivElement>;
export declare const PromptInputTabItem: ({ className, ...props }: PromptInputTabItemProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputCommandProps = ComponentProps<typeof Command>;
export declare const PromptInputCommand: ({ className, ...props }: PromptInputCommandProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputCommandInputProps = ComponentProps<typeof CommandInput>;
export declare const PromptInputCommandInput: ({ className, ...props }: PromptInputCommandInputProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputCommandListProps = ComponentProps<typeof CommandList>;
export declare const PromptInputCommandList: ({ className, ...props }: PromptInputCommandListProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputCommandEmptyProps = ComponentProps<typeof CommandEmpty>;
export declare const PromptInputCommandEmpty: ({ className, ...props }: PromptInputCommandEmptyProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputCommandGroupProps = ComponentProps<typeof CommandGroup>;
export declare const PromptInputCommandGroup: ({ className, ...props }: PromptInputCommandGroupProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputCommandItemProps = ComponentProps<typeof CommandItem>;
export declare const PromptInputCommandItem: ({ className, ...props }: PromptInputCommandItemProps) => import("react/jsx-runtime").JSX.Element;
export type PromptInputCommandSeparatorProps = ComponentProps<typeof CommandSeparator>;
export declare const PromptInputCommandSeparator: ({ className, ...props }: PromptInputCommandSeparatorProps) => import("react/jsx-runtime").JSX.Element;
export {};
